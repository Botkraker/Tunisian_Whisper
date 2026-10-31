"""Live Tunisian Derja transcription on faster-whisper.

    python live.py                    # microphone, tier picked from the hardware
    python live.py --file sound.wav   # replay a file in real time
    python live.py --tier lite        # force the CPU tier

Lite runs one small Derja model on the CPU. Pro (an NVIDIA GPU with at least 4 GB) keeps the small model
for the live partials and decodes each finished turn with a large-v3 Derja model. Model loading and decoding
live here so that bakeoff.py scores exactly what the app ships.
"""

import argparse
import ctypes
import importlib
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from derja_text import is_degenerate

SR = 16000
FRAME = 512  # samples per Silero VAD frame (32 ms)


@dataclass(frozen=True)
class StreamConfig:
    partial_every_s: float = 0.5
    min_silence_s: float = 0.5  # the pause that ends a turn; it is part of the final-line latency
    preroll_s: float = 0.3
    tail_pad_s: float = 0.2
    min_speech_s: float = 0.25
    max_turn_s: float = 25.0  # cut long turns before Whisper's 30 s window
    vad_threshold: float = 0.5
    max_partial_s: float = 1.0


@dataclass(frozen=True)
class Event:
    kind: str  # "partial" or "final"
    stable: str
    unstable: str = ""
    start_s: float = 0.0
    end_s: float = 0.0


class LocalAgreement:
    """LocalAgreement-2 (ufal/whisper_streaming): a word is confirmed once two decodes in a row agree on it.

    Confirmed words never change, so the live line only flickers after its confirmed part.
    """

    def __init__(self) -> None:
        self._confirmed: list[str] = []
        self._previous: list[str] = []

    def update(self, words: list[str]) -> tuple[list[str], list[str]]:
        agreed = []
        for before, now in zip(self._previous, words):
            if before != now:
                break
            agreed.append(now)
        if len(agreed) > len(self._confirmed) and agreed[: len(self._confirmed)] == self._confirmed:
            self._confirmed = agreed
        self._previous = words
        return list(self._confirmed), words[len(self._confirmed) :]


class SileroVad:
    """The Silero VAD v6 model bundled with faster-whisper, run on consecutive frames with its state carried over."""

    CONTEXT = 64

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        contexts = np.vstack([self._context, frames[:-1, -self.CONTEXT :]])
        self._context = frames[-1, -self.CONTEXT :].copy()
        inputs = {"input": np.hstack([contexts, frames]).astype(np.float32), "h": self._h, "c": self._c}
        probabilities, self._h, self._c = self._session.run(None, inputs)
        return probabilities


class TurnTranscriber:
    """Turns a 16 kHz mono float32 stream into partial and final Events.

    VAD frames open a turn at speech onset and close it after min_silence_s of silence. While a turn is open,
    its audio is re-decoded by decode_partial every partial_every_s, and LocalAgreement decides which words
    are confirmed. A closed turn is decoded once more by decode_final. Each event goes to on_event the moment
    it exists, so a final is never held back by a later decode in the same feed() call.
    """

    def __init__(
        self,
        decode_partial: Callable[[np.ndarray], str],
        decode_final: Callable[[np.ndarray], str],
        vad: Callable[[np.ndarray], np.ndarray],
        config: StreamConfig = StreamConfig(),
        on_event: Callable[[Event], None] | None = None,
    ):
        self._decode_partial = decode_partial
        self._decode_final = decode_final
        self._vad = vad
        self._config = config
        self._on_event = on_event
        self._preroll = round(config.preroll_s * SR)
        self._tail_pad = round(config.tail_pad_s * SR)
        self._max_turn = round(config.max_turn_s * SR)
        self._partial_every = round(config.partial_every_s * SR)
        self._min_silence_frames = round(config.min_silence_s * SR / FRAME)
        self._min_speech_frames = round(config.min_speech_s * SR / FRAME)
        self._cut_window = round(2.0 * SR / FRAME)  # a too-long turn is cut at its quietest frame in the last 2 s
        self._framed = 0  # stream samples that have been through the VAD
        self._unframed = np.zeros(0, np.float32)  # tail samples waiting for a full VAD frame
        self._recent = np.zeros(0, np.float32)  # idle audio kept as preroll
        self._turn: list[np.ndarray] | None = None
        self.partials = True  # switched off for good once a partial decode takes longer than max_partial_s

    def feed(self, chunk: np.ndarray) -> list[Event]:
        samples = np.concatenate([self._unframed, np.asarray(chunk, np.float32).reshape(-1)])
        count = len(samples) // FRAME
        self._unframed = samples[count * FRAME :]
        events: list[Event] = []
        if count:
            frames = samples[: count * FRAME].reshape(count, FRAME)
            for frame, probability in zip(frames, self._vad(frames)):
                events += self._step(frame, probability)
        # At most one partial per call, so catching up after a slow decode never queues more decodes, and none
        # once a pause has begun (3 frames, about 0.1 s): it would only hold up the final.
        if (
            self.partials
            and self._turn is not None
            and self._trailing_silence < 3
            and self._framed - self._last_partial >= self._partial_gap
        ):
            events += self._partial()
        return events

    def flush(self) -> list[Event]:
        events = []
        if len(self._unframed):
            frame = np.pad(self._unframed, (0, FRAME - len(self._unframed)))
            self._unframed = self._unframed[:0]
            events += self._step(frame, self._vad(frame[None])[0])
        if self._turn is not None:
            events += self._close_turn()
        return events

    def _step(self, frame: np.ndarray, probability: float) -> list[Event]:
        self._framed += FRAME
        if self._turn is None:
            if probability < self._config.vad_threshold:
                self._recent = _tail(np.concatenate([self._recent, frame]), self._preroll)
                return []
            self._start_turn(self._recent, start=self._framed - FRAME - len(self._recent))
        self._add_frame(frame, probability)
        if self._trailing_silence >= self._min_silence_frames:
            return self._close_turn()
        if self._turn_samples >= self._max_turn:
            return self._cut_turn()
        return []

    def _start_turn(self, preroll: np.ndarray, start: int) -> None:
        self._turn = [preroll]
        self._preroll_samples = len(preroll)
        self._probabilities: list[float] = []  # one per frame after the preroll
        self._turn_samples = len(preroll)
        self._turn_start = start
        self._recent = np.zeros(0, np.float32)
        self._speech_frames = 0
        self._trailing_silence = 0
        self._last_partial = self._framed
        self._partial_gap = self._partial_every
        self._agreement = LocalAgreement()

    def _add_frame(self, frame: np.ndarray, probability: float) -> None:
        self._turn.append(frame)
        self._probabilities.append(probability)
        self._turn_samples += len(frame)
        if probability >= self._config.vad_threshold - 0.15:  # Silero's hysteresis keeps soft speech in the turn
            self._speech_frames += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

    def _close_turn(self) -> list[Event]:
        audio = np.concatenate(self._turn)
        silence = self._trailing_silence * FRAME
        end = self._framed - silence
        self._turn = None
        self._recent = _tail(audio, self._preroll)
        return self._final(audio[: len(audio) - silence + min(silence, self._tail_pad)], end)

    def _cut_turn(self) -> list[Event]:
        window = self._probabilities[-self._cut_window :]
        quietest = len(self._probabilities) - 1 - int(np.argmin(window[::-1]))  # the latest of the quietest frames
        cut = self._preroll_samples + (quietest + 1) * FRAME
        audio = np.concatenate(self._turn)
        carried = list(zip(audio[cut:].reshape(-1, FRAME), self._probabilities[quietest + 1 :]))
        events = self._final(audio[:cut], end=self._turn_start + cut)
        self._start_turn(np.zeros(0, np.float32), start=self._turn_start + cut)  # the speaker is still talking
        for frame, probability in carried:
            self._add_frame(frame, probability)
        return events

    def _final(self, audio: np.ndarray, end: int) -> list[Event]:
        if self._speech_frames < self._min_speech_frames:
            return []
        text = self._decode_final(audio)
        if is_degenerate(text):
            return []
        return self._emit(Event("final", text, "", self._turn_start / SR, end / SR))

    def _emit(self, event: Event) -> list[Event]:
        if self._on_event is not None:
            self._on_event(event)
        return [event]

    def _partial(self) -> list[Event]:
        started = time.perf_counter()
        text = self._decode_partial(np.concatenate(self._turn))
        elapsed = time.perf_counter() - started
        if elapsed > self._config.max_partial_s:
            self.partials = False  # partials cannot be live on this machine, so leave all decode time to finals
        # keep partial decoding below real time on slow machines, so finals are never queued behind partials
        self._partial_gap = max(self._partial_every, round(1.5 * elapsed * SR))
        self._last_partial = self._framed
        if is_degenerate(text):
            return []
        confirmed, tentative = self._agreement.update(text.split())
        return self._emit(Event("partial", " ".join(confirmed), " ".join(tentative), self._turn_start / SR, self._framed / SR))


def _tail(audio: np.ndarray, samples: int) -> np.ndarray:
    return audio[max(len(audio) - samples, 0) :]


def load_model(path: str, device: str = "cpu", compute_type: str = "int8", cpu_threads: int = 4):
    if device == "cuda":
        _preload_pip_cuda_libraries()
    from faster_whisper import WhisperModel

    return WhisperModel(path, device=device, compute_type=compute_type, cpu_threads=cpu_threads)


def decode(model, audio, beam_size: int, language: str = "ar") -> str:
    segments, _ = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=False,
        no_speech_threshold=None,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def skip_failures(decoder: Callable[[np.ndarray], str]) -> Callable[[np.ndarray], str]:
    """Logs a failed decode (such as CUDA running out of memory on a long final) and drops it, so the session goes on."""

    def safe(audio: np.ndarray) -> str:
        try:
            return decoder(audio)
        except RuntimeError as error:
            print(f"\ndecode failed and was skipped: {error}", file=sys.stderr)
            return ""

    return safe


def _preload_pip_cuda_libraries() -> None:
    """CTranslate2 dlopens cuBLAS 12 and cuDNN 9 by name; loading the pip nvidia-* copies first avoids LD_LIBRARY_PATH."""
    for package, names in (("nvidia.cublas", ("libcublasLt.so.12", "libcublas.so.12")), ("nvidia.cudnn", ("libcudnn.so.9",))):
        try:
            lib_dir = Path(next(iter(importlib.import_module(package).__path__))) / "lib"
        except ImportError:
            continue
        for name in names:
            if (lib_dir / name).exists():
                ctypes.CDLL(str(lib_dir / name), mode=ctypes.RTLD_GLOBAL)


ROOT = Path(__file__).resolve().parent
LITE_MODEL = "TheBiggyBang/whisper-small-tunisian-derja-ct2"
# Chosen after the bake-off (see bench/results/REPORT.md). oddadmix/Whisperv3-tunisian-codeswitch states no
# licence: ask its author before publishing anything built on it. montachouig-ct2 (Apache-2.0) is the fallback.
PRO_MODEL = str(ROOT / "models" / "oddadmix-ct2")
FINAL_BEAM = 3
PARTIALS_OFF = "Partials are off: one takes over a second to decode here, so only final lines are shown."


@dataclass(frozen=True)
class Tier:
    partial_model: str
    final_model: str
    device: str
    compute_type: str


TIERS = {
    "lite": Tier(LITE_MODEL, LITE_MODEL, "cpu", "int8"),
    # int8_float32 keeps GTX 16xx cards off fp16, where cuDNN convolutions return NaN
    "pro": Tier(LITE_MODEL, PRO_MODEL, "cuda", "int8_float32"),
}


def detect_tier() -> str:
    """Pro needs an NVIDIA GPU with at least 4 GB and the Pro model on disk; anything else runs Lite."""
    import ctranslate2

    if ctranslate2.get_cuda_device_count() == 0 or not Path(PRO_MODEL).is_dir():
        return "lite"
    try:
        query = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        total_mb = int(subprocess.run(query, capture_output=True, text=True, check=True).stdout.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return "lite"
    return "pro" if total_mb >= 4000 else "lite"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="replay this audio file in real time instead of listening to the microphone")
    parser.add_argument("--tier", choices=("auto", "lite", "pro"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1), help="CPU threads")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic finals must print even where the console default is cp1252

    name = detect_tier() if args.tier == "auto" else args.tier
    if name == "pro" and not Path(PRO_MODEL).is_dir():
        sys.exit(f"The Pro model is missing ({PRO_MODEL}). Convert it with convert_models.py, or use --tier lite.")
    try:
        partial_model, final_model = _load_tier(TIERS[name], args.threads)
    except (RuntimeError, OSError, ValueError) as error:
        if args.tier != "auto" or name == "lite":
            raise
        print(f"The Pro tier did not load ({error}), so Lite is used instead.", file=sys.stderr)
        name = "lite"
        partial_model, final_model = _load_tier(TIERS[name], args.threads)
    tier = TIERS[name]
    console = Console()
    engine = TurnTranscriber(
        skip_failures(lambda audio: decode(partial_model, audio, beam_size=1)),
        skip_failures(lambda audio: decode(final_model, audio, beam_size=FINAL_BEAM)),
        SileroVad(),
        on_event=console.show,
    )
    print(f"{name} tier on {tier.device} ({tier.compute_type}), finals from {tier.final_model}", file=sys.stderr)
    engine.partials = _partials_are_fast(partial_model)
    if not engine.partials:
        print(PARTIALS_OFF, file=sys.stderr)
    finished = replay(engine, console, args.file) if args.file else listen(engine, console)
    for kind, label in (("final", "final line after the pause"), ("partial", "partial text lag")):
        if console.lags[kind]:
            lags = console.lags[kind]
            print(f"{label}: median {statistics.median(lags):.2f} s, max {max(lags):.2f} s", file=sys.stderr)
    if not finished:
        sys.exit(1)


def _partials_are_fast(model) -> bool:
    """Times one more warm-up decode, so a slow machine starts with partials off instead of stalling its first turn."""
    started = time.perf_counter()
    decode(model, np.zeros(SR, np.float32), beam_size=1)
    return time.perf_counter() - started <= StreamConfig().max_partial_s


def _load_tier(tier: Tier, threads: int):
    partial = load_model(tier.partial_model, tier.device, tier.compute_type, threads)
    final = (
        partial
        if tier.final_model == tier.partial_model
        else load_model(tier.final_model, tier.device, tier.compute_type, threads)
    )
    for model in {id(partial): partial, id(final): final}.values():
        decode(model, np.zeros(SR, np.float32), beam_size=1)  # warm-up, so the first real decode is not the slowest
    return partial, final


def replay(engine: TurnTranscriber, console: "Console", path: str, chunk_s: float = 0.1) -> bool:
    from faster_whisper import decode_audio

    # a second of trailing silence lets the last turn end with a pause, as it would live
    audio = np.concatenate([decode_audio(path, sampling_rate=SR), np.zeros(SR, np.float32)])
    step = round(chunk_s * SR)
    console.start = time.perf_counter()
    fed = 0
    try:
        while fed < len(audio):
            # feed everything that has "arrived" by now, as the microphone queue does after a slow decode
            until = min(len(audio), max(fed + step, round((time.perf_counter() - console.start) * SR)))
            time.sleep(max(0.0, console.start + until / SR - time.perf_counter()))
            _feed(engine, console, audio[fed:until])
            fed = until
    except KeyboardInterrupt:
        pass
    engine.flush()
    return True


def listen(engine: TurnTranscriber, console: "Console", chunk_s: float = 0.1) -> bool:
    """Transcribes the microphone until Ctrl+C; returns False if the microphone stopped by itself."""
    import queue

    try:
        import sounddevice
    except OSError:
        sys.exit("The microphone needs PortAudio. On Linux: sudo apt install libportaudio2")
    chunks: queue.Queue[np.ndarray] = queue.Queue()
    try:
        stream = sounddevice.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocksize=round(chunk_s * SR),
            callback=lambda data, frames, when, status: chunks.put(data[:, 0].copy()),
        )
    except sounddevice.PortAudioError as error:
        sys.exit(f"Could not open the microphone: {error}")
    print("Listening. Press Ctrl+C to stop.", file=sys.stderr)
    stopped_by_user = False
    with stream:
        console.start = time.perf_counter()
        try:
            while stream.active:
                try:
                    audio = [chunks.get(timeout=1.0)]
                except queue.Empty:
                    continue
                while not chunks.empty():  # catch up after a slow decode
                    audio.append(chunks.get_nowait())
                _feed(engine, console, np.concatenate(audio))
        except KeyboardInterrupt:
            stopped_by_user = True
    try:  # transcribe what was still queued, then the open turn; a second Ctrl+C skips this
        leftover = [chunks.get_nowait() for _ in range(chunks.qsize())]
        if leftover:
            engine.feed(np.concatenate(leftover))
        engine.flush()
    except KeyboardInterrupt:
        pass
    if not stopped_by_user:
        console.note("The microphone stopped delivering audio.")
    return stopped_by_user


def _feed(engine: TurnTranscriber, console: "Console", audio: np.ndarray) -> None:
    had_partials = engine.partials
    engine.feed(audio)
    if had_partials and not engine.partials:
        console.note(PARTIALS_OFF)


class Console:
    """Shows each event the moment the engine emits it, and records how long after its audio it appeared."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.lags: dict[str, list[float]] = {"partial": [], "final": []}
        self._terminal = sys.stdout.isatty()

    def show(self, event: Event) -> None:
        lag = time.perf_counter() - self.start - event.end_s
        self.lags[event.kind].append(lag)
        span = f"[{event.start_s:6.1f}–{event.end_s:6.1f} s]"
        if event.kind == "partial" and self._terminal:
            width = shutil.get_terminal_size().columns - 3  # "… " plus one spare column, so the line never wraps
            rtl = is_arabic(event.stable + event.unstable)
            sys.stdout.write("\r\033[K… " + partial_line(event.stable, event.unstable, width, rtl=rtl))
        elif event.kind == "final" and self._terminal:
            sys.stdout.write(f"\r\033[K{span} {terminal_text(event.stable)}  \033[2m(+{lag:.1f} s)\033[0m\n")
        elif event.kind == "final":
            sys.stdout.write(f"{span} {event.stable}  (+{lag:.1f} s)\n")
        sys.stdout.flush()

    def note(self, message: str) -> None:
        if self._terminal:
            sys.stdout.write("\r\033[K")  # clear the partial line first, so the note does not strand it
            sys.stdout.flush()
        print(message, file=sys.stderr)


def terminal_text(text: str) -> str:
    """Text as a left-to-right terminal must draw it: Arabic letters joined, and the line laid out right to left."""
    try:
        import arabic_reshaper
        from bidi import get_display
    except ImportError:
        return text
    return get_display(arabic_reshaper.reshape(text))


def is_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "ۿ" for ch in text)


def partial_line(stable: str, unstable: str, width: int, rtl: bool = False) -> str:
    """The live partial, cut to fit the terminal (a wrapped line cannot be redrawn in place), oldest words first out.

    With rtl the line is laid out for Arabic: the tentative, newest words sit on the left.
    """
    unstable = unstable[max(len(unstable) - width, 0) :]
    stable = stable[max(len(stable) - max(width - len(unstable) - 1, 0), 0) :]
    dim = f"\033[2m{terminal_text(unstable) if rtl else unstable}\033[0m" if unstable else ""
    parts = (dim, terminal_text(stable)) if rtl else (stable, dim)
    return " ".join(part for part in parts if part)


if __name__ == "__main__":
    main()
