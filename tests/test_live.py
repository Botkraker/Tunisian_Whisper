import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from live import (
    FRAME,
    LITE_MODEL,
    SR,
    LocalAgreement,
    SileroVad,
    StreamConfig,
    TurnTranscriber,
    partial_line,
    skip_failures,
    terminal_text,
)

CHUNK = 3 * FRAME  # 96 ms; frame-aligned, so every chunk is exactly three VAD frames
ROOT = Path(__file__).resolve().parent.parent
SOUND = ROOT / "sound.wav"


def speech(chunks: int) -> list[np.ndarray]:
    return [np.full(CHUNK, 0.5, np.float32)] * chunks


def soft_speech(chunks: int) -> list[np.ndarray]:
    return [np.full(CHUNK, 0.4, np.float32)] * chunks


def silence(chunks: int) -> list[np.ndarray]:
    return [np.zeros(CHUNK, np.float32)] * chunks


def fake_vad(frames: np.ndarray) -> np.ndarray:
    """Speech probability of a frame = its sample value, so the test audio spells out what the VAD sees."""
    return frames.mean(axis=1)


class ScriptedDecoder:
    """Returns its outputs in order (repeating the last one) and records the length of every audio it gets."""

    def __init__(self, *outputs: str):
        self.outputs = outputs
        self.calls: list[int] = []

    def __call__(self, audio: np.ndarray) -> str:
        self.calls.append(len(audio))
        return self.outputs[min(len(self.calls), len(self.outputs)) - 1]


SETTINGS = {"preroll_s": 0.0, "tail_pad_s": 0.096, "min_silence_s": 0.48, "min_speech_s": 0.192, "partial_every_s": 100}


def run(chunks, partial=None, final=None, **config):
    partial = partial or ScriptedDecoder("partial")
    final = final or ScriptedDecoder("final text")
    engine = TurnTranscriber(partial, final, fake_vad, StreamConfig(**{**SETTINGS, **config}))
    events = [event for chunk in chunks for event in engine.feed(chunk)]
    return events + engine.flush(), partial, final


def test_local_agreement_waits_for_two_decodes_before_confirming():
    assert LocalAgreement().update(["a", "b"]) == ([], ["a", "b"])


def test_local_agreement_confirms_the_prefix_two_decodes_share():
    agreement = LocalAgreement()
    agreement.update(["a", "x"])
    assert agreement.update(["a", "y", "z"]) == (["a"], ["y", "z"])


def test_local_agreement_never_retracts_confirmed_words():
    agreement = LocalAgreement()
    agreement.update(["a", "b"])
    agreement.update(["a", "b", "c"])
    assert agreement.update(["a", "x"]) == (["a", "b"], [])


def test_silence_is_never_decoded():
    events, partial, final = run(silence(50), partial_every_s=0.48)
    assert events == []
    assert partial.calls == [] and final.calls == []


def test_a_pause_ends_the_turn_with_one_final_decode_of_the_speech():
    events, _, final = run(speech(10) + silence(6))
    assert [(event.kind, event.stable) for event in events] == [("final", "final text")]
    assert final.calls == [(30 + 3) * FRAME]  # the speech plus the tail pad, not the whole pause


def test_preroll_keeps_audio_from_just_before_the_onset():
    _, _, final = run(silence(5) + speech(10) + silence(6), preroll_s=0.192)
    assert final.calls == [(6 + 30 + 3) * FRAME]


def test_final_event_spans_the_turn_from_preroll_to_the_last_speech_frame():
    events, _, _ = run(silence(5) + speech(10) + silence(6), preroll_s=0.192)
    assert (events[0].start_s, events[0].end_s) == pytest.approx((0.288, 1.44))


def test_partials_show_words_once_two_decodes_agree():
    partial = ScriptedDecoder("a", "a b", "a b c", "a b c d")
    events, _, _ = run(speech(25) + silence(6), partial=partial, partial_every_s=0.48)
    partials = [(event.stable, event.unstable) for event in events if event.kind == "partial"]
    assert partials == [("", "a"), ("a", "b"), ("a b", "c"), ("a b c", "d")]


def test_soft_speech_inside_a_turn_does_not_end_it():
    _, _, final = run(speech(5) + soft_speech(10) + silence(6))
    assert final.calls == [(15 + 30 + 3) * FRAME]


def test_soft_speech_alone_does_not_open_a_turn():
    events, _, final = run(soft_speech(20) + silence(6))
    assert events == [] and final.calls == []


def test_no_partial_is_decoded_once_the_pause_has_begun():
    partial = ScriptedDecoder("a")
    run(speech(20) + silence(6), partial=partial, partial_every_s=0.48)
    assert len(partial.calls) == 3  # the fourth would land in the pause and hold up the final


def test_each_event_reaches_on_event_before_the_next_decode_starts():
    delivered: list = []
    seen_by_decodes: list[int] = []

    def final(audio):
        seen_by_decodes.append(len(delivered))
        return "final text"

    two_turns = np.concatenate(speech(10) + silence(7) + speech(10) + silence(7))  # one catch-up chunk
    engine = TurnTranscriber(ScriptedDecoder("p"), final, fake_vad, StreamConfig(**SETTINGS), on_event=delivered.append)
    engine.feed(two_turns)
    assert seen_by_decodes == [0, 1]  # the first final was out before the second decode began
    assert [event.kind for event in delivered] == ["final", "final"]


def test_partials_stop_once_a_partial_decode_is_too_slow():
    def slow(audio):
        time.sleep(0.03)
        return "a"

    events, _, final = run(speech(30) + silence(6), partial=slow, partial_every_s=0.48, max_partial_s=0.01)
    assert [event.kind for event in events] == ["partial", "final"]  # without the limit: six partials
    assert len(final.calls) == 1


def test_a_blip_shorter_than_min_speech_is_dropped():
    events, _, final = run(speech(1) + silence(6))
    assert events == [] and final.calls == []


def test_flush_finalizes_the_open_turn_at_end_of_stream():
    events, _, final = run(speech(10))
    assert [(event.kind, event.stable) for event in events] == [("final", "final text")]
    assert final.calls == [30 * FRAME]


def test_a_long_turn_is_cut_before_whispers_30_s_window():
    _, _, final = run(speech(30), max_turn_s=1.44)
    assert final.calls == [45 * FRAME, 45 * FRAME]


def test_turns_do_not_depend_on_how_the_stream_is_chunked():
    stream = np.concatenate(speech(10) + silence(7) + speech(10) + silence(7))  # two sentences, 0.67 s pauses
    small = run([stream[i : i + 1600] for i in range(0, len(stream), 1600)])
    large = run([stream[i : i + 16000] for i in range(0, len(stream), 16000)])  # what a slow decode's catch-up feeds
    assert small[2].calls == large[2].calls == [33 * FRAME, 33 * FRAME]


def test_flush_keeps_the_audio_after_the_last_full_frame():
    _, _, final = run(speech(10) + [np.full(100, 0.5, np.float32)])
    assert final.calls[0] >= 30 * FRAME + 100


def test_a_long_turn_is_cut_at_its_quietest_frame_and_the_rest_carries_over():
    dip = np.full(FRAME, 0.36, np.float32)  # still speech inside a turn, but the quietest frame
    _, _, final = run(speech(10) + [dip, np.full(20 * FRAME, 0.5, np.float32)], max_turn_s=1.44)
    assert final.calls == [31 * FRAME, 20 * FRAME]


def test_a_failing_decode_is_reported_and_skipped(capsys):
    def out_of_memory(audio):
        raise RuntimeError("CUDA failed with error out of memory")

    assert skip_failures(out_of_memory)(np.zeros(SR, np.float32)) == ""
    assert "out of memory" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("stable", "unstable", "width", "expected"),
    [
        ("a b", "c", 80, "a b \033[2mc\033[0m"),
        ("a b c d", "e f", 7, "c d \033[2me f\033[0m"),  # too wide: the oldest words go
        ("a b", "", 80, "a b"),
        ("", "a b", 80, "\033[2ma b\033[0m"),
    ],
)
def test_partial_line_keeps_the_newest_words_within_the_terminal_width(stable, unstable, width, expected):
    assert partial_line(stable, unstable, width) == expected


@pytest.mark.parametrize(
    ("logical", "on_screen"),
    [
        ("يا شبينا", "انيبش اي"),  # a terminal draws left to right, so the line is laid out reversed
        ("bonjour ça va", "bonjour ça va"),  # Latin text is left alone
        ("يا bonjour", "bonjour اي"),  # a French word stays readable inside an Arabic line
    ],
)
def test_terminal_text_lays_out_arabic_right_to_left(logical, on_screen):
    import unicodedata

    # NFKC undoes the joined letter shapes, so only the order is compared
    assert unicodedata.normalize("NFKC", terminal_text(logical)) == on_screen


def test_degenerate_output_is_not_shown():
    events, _, _ = run(speech(10) + silence(6), final=ScriptedDecoder("!!!!"))
    assert events == []


def test_streaming_vad_matches_faster_whispers_batch_vad():
    from faster_whisper import decode_audio
    from faster_whisper.vad import get_vad_model

    audio = decode_audio(str(SOUND), sampling_rate=SR)
    frames = audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME)
    expected = get_vad_model()(frames.reshape(-1).copy())
    vad = SileroVad()
    streamed = np.concatenate([vad(frames[i : i + 7]) for i in range(0, len(frames), 7)])
    # faster-whisper zeroes the tail of the last frame before scoring it, so that frame is left out
    np.testing.assert_allclose(streamed[:-1], expected[:-1], atol=1e-5)


def test_cli_transcribes_a_replayed_file(tmp_path):
    from faster_whisper import decode_audio
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(LITE_MODEL, "model.bin"), str):
        pytest.skip("the Lite model is not downloaded yet")
    clip = tmp_path / "clip.wav"
    audio = decode_audio(str(SOUND), sampling_rate=SR)[: 4 * SR]  # starts with "يا شبينا"
    with wave.open(str(clip), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    command = [sys.executable, str(ROOT / "live.py"), "--file", str(clip), "--tier", "lite"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert re.search(r"^\[\s*[\d.]+–\s*[\d.]+ s\] \S", result.stdout, re.MULTILINE), result.stdout
    assert "\033" not in result.stdout  # piped output gets plain final lines, no terminal codes
