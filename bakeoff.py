"""Bake-off harness: accuracy and speed of CTranslate2 Whisper models.

    python bakeoff.py prepare                                   # 100 LinTO + 100 FARUK test clips
    python bakeoff.py sanity   --model DIR                      # sound.wav chunk 0 must read "يا شبينا"
    python bakeoff.py accuracy --model DIR --name NAME [--device cuda]
    python bakeoff.py speed    --model DIR --name NAME [--device cpu --threads 4]
    python bakeoff.py report                                    # bench/results/REPORT.md

--model takes a CTranslate2 directory or a Hugging Face repo id. Every model gets the same decoding:
greedy, language="ar", temperature 0, no timestamps. CPU runs use int8 and GPU runs int8_float32,
which keeps the GTX 1650 off the fp16 path.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

from derja_text import cer, is_degenerate, normalize, wer
from live import FRAME, SR, StreamConfig, decode, load_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "bench" / "data"
RESULTS_DIR = ROOT / "bench" / "results"
MANIFEST = DATA_DIR / "manifest.jsonl"
EAR_TEST = ROOT / "sound.wav"

LINTO_SUBSETS = ("AmenyKH", "ApprendreLeTunisien", "OneStory", "TunSwitchCS", "TunSwitchTO", "Youtube_TNScrapped_V1")
SOURCES = {
    "linto": (
        "linagora/linto-dataset-audio-ar-tn",
        [f"data/{subset}/test/test-00000-of-00001.parquet" for subset in LINTO_SUBSETS],
        "transcript",
    ),
    "faruk": ("FARUKxAUTO/tunisian-asr-cleaned", ["data/test-00000-of-00002.parquet"], "sentence"),
}
CLIPS_PER_SOURCE = 100
CLIP_SECONDS = (0.5, 30.0)
SPEED_CLIP_SECONDS = 5.0
LATENCY_BUDGET_S = 1.5  # the final line must appear within 1.5 s after the speaker pauses
PIPELINE_LAG_S = 0.1 + FRAME / SR  # one microphone chunk plus one VAD frame, before the pause is even counted


def prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    if MANIFEST.exists() and not args.force:
        sys.exit("A bench set already exists, and results only compare on the same set. Use --force to replace it.")
    MANIFEST.unlink(missing_ok=True)  # so a failed re-run cannot leave the old manifest paired with new clips
    clips_dir = DATA_DIR / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    download_dir = DATA_DIR / ".download"  # kept until the bench set is complete, so a failed run resumes
    rng = np.random.default_rng(args.seed)
    manifest, snapshot = [], {}
    for source, (repo, files, text_column) in SOURCES.items():
        revision = HfApi().dataset_info(repo).sha
        snapshot[source] = {"repo": repo, "revision": revision, "files": files}
        rows = []
        for name in files:
            path = hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=download_dir)
            refs = pq.read_table(path, columns=[text_column]).column(text_column).to_pylist()
            rows += [(path, name, row, ref) for row, ref in enumerate(refs) if ref and normalize(ref)]
        picked = _sample_clips(rows, rng.permutation(len(rows)), CLIPS_PER_SOURCE)
        for n, ((_, name, row, ref), audio) in enumerate(picked, start=1):
            clip_id = f"{source}-{n:03d}"
            _write_wav(clips_dir / f"{clip_id}.wav", audio)
            manifest.append(
                {
                    "id": clip_id,
                    "source": source,
                    "subset": name.split("/")[1] if source == "linto" else Path(name).stem,
                    "origin": f"{repo}/{name}#{row}",
                    "audio": f"clips/{clip_id}.wav",
                    "duration": round(len(audio) / SR, 3),
                    "ref": ref,
                }
            )
        print(f"{source}: {len(picked)} clips sampled from {len(rows)}")
    MANIFEST.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest), encoding="utf-8")
    (DATA_DIR / "snapshot.json").write_text(json.dumps({"seed": args.seed, "sources": snapshot}, indent=2))
    shutil.rmtree(download_dir)


def _sample_clips(rows: list[tuple], order: np.ndarray, count: int) -> list[tuple]:
    picked = []
    for start in range(0, len(order), count):
        window = [rows[i] for i in order[start : start + count]]
        for row, audio in zip(window, _read_audio(window)):
            if CLIP_SECONDS[0] <= len(audio) / SR <= CLIP_SECONDS[1]:
                picked.append((row, audio))
        if len(picked) >= count:
            return picked[:count]
    raise RuntimeError(f"only {len(picked)} usable clips, wanted {count}")


def _read_audio(rows: list[tuple]) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    wanted_by_file: dict[str, set[int]] = {}
    for path, _, row, _ in rows:
        wanted_by_file.setdefault(path, set()).add(row)
    audio = {}
    for path, wanted in wanted_by_file.items():
        parquet = pq.ParquetFile(path)
        first = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            needed = [row for row in sorted(wanted) if first <= row < first + size]
            if needed:
                column = parquet.read_row_group(group, columns=["audio"]).column("audio")
                for row in needed:
                    encoded = column[row - first].as_py()["bytes"]
                    audio[path, row] = decode_audio(io.BytesIO(encoded), sampling_rate=SR)
            first += size
    return [audio[path, row] for path, _, row, _ in rows]


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(pcm.tobytes())


def sanity(args: argparse.Namespace) -> None:
    model = _load(args)
    hyp = decode(model, decode_audio(str(EAR_TEST), sampling_rate=SR)[: 2 * SR], beam_size=1)
    if normalize(hyp) == normalize(args.expect):
        verdict = "PASS"
    elif normalize(hyp).replace(" ", "") == normalize(args.expect).replace(" ", ""):
        # This clip sits between two word splits of the same letters ("يا شبينا" / "ياش بينا"); even the
        # Hugging Face fp16 reference flips with the audio resampler, so only the letters can be checked.
        verdict = "PASS (same letters, different word split)"
    else:
        verdict = "FAIL"
    print(f"{verdict}: sound.wav chunk 0 -> {hyp!r} (expected {args.expect!r})")
    sys.exit(0 if verdict.startswith("PASS") else 1)


def accuracy(args: argparse.Namespace) -> None:
    clips = _load_manifest()
    model = _load(args)
    decode(model, _load_clip(clips[0]), beam_size=1)  # warm-up, not timed
    rows = []
    for clip in clips:
        audio = _load_clip(clip)
        start = time.perf_counter()
        hyp = decode(model, audio, beam_size=1)
        rows.append({"id": clip["id"], "hyp": hyp, "seconds": round(time.perf_counter() - start, 4)})
    ear = decode_audio(str(EAR_TEST), sampling_rate=SR)
    segments, _ = model.transcribe(
        ear, language="ar", beam_size=1, temperature=0.0, vad_filter=True, condition_on_previous_text=False
    )
    result = {
        **_run_info(args),
        "summary": _score(clips, rows),
        "ear_test": {
            "chunk0": decode(model, ear[: 2 * SR], beam_size=1),
            "sound.wav": " ".join(segment.text.strip() for segment in segments),
        },
        "clips": rows,
    }
    _save("accuracy", args, result)
    for group, scores in result["summary"].items():
        print(f"{group:6} {scores}")
    print(f"ear test: {result['ear_test']['sound.wav']}")


def _score(clips: list[dict], rows: list[dict]) -> dict:
    summary = {}
    for group in ("linto", "faruk", "all"):
        pairs = [(clip, row) for clip, row in zip(clips, rows, strict=True) if group in ("all", clip["source"])]
        refs = [normalize(clip["ref"]) for clip, _ in pairs]
        hyps = [normalize(row["hyp"]) for _, row in pairs]
        summary[group] = {
            "clips": len(pairs),
            "wer": round(100 * wer(refs, hyps), 2),
            "cer": round(100 * cer(refs, hyps), 2),
            "rtf": round(sum(row["seconds"] for _, row in pairs) / sum(clip["duration"] for clip, _ in pairs), 4),
            "degenerate": sum(is_degenerate(row["hyp"]) for _, row in pairs),
        }
    return summary


def speed(args: argparse.Namespace) -> None:
    clip = min(_load_manifest(), key=lambda entry: abs(entry["duration"] - SPEED_CLIP_SECONDS))
    audio = _load_clip(clip)
    model = _load(args)
    for beam_size in (1, 3):
        decode(model, audio, beam_size=beam_size)  # warm-up, not timed
    partial = _time_runs(lambda: decode(model, audio, beam_size=1), args.repeats)
    final = _time_runs(lambda: decode(model, audio, beam_size=3), args.repeats)
    after_pause = final["median"] + StreamConfig().min_silence_s + PIPELINE_LAG_S
    result = {
        **_run_info(args),
        "clip": clip["id"],
        "clip_seconds": clip["duration"],
        "partial_s": partial,
        "final_s": final,
        "final_after_pause_s": round(after_pause, 3),
        "rss_mb": _rss_mb(),
        "vram_mb": _vram_mb() if args.device == "cuda" else None,
        "passes": after_pause <= LATENCY_BUDGET_S,
    }
    _save("speed", args, result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


def _time_runs(run, repeats: int) -> dict:
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        seconds.append(time.perf_counter() - start)
    return {"median": round(statistics.median(seconds), 3), "max": round(max(seconds), 3)}


def _rss_mb() -> int:
    import psutil

    return round(psutil.Process().memory_info().rss / 2**20)


def _vram_mb() -> int | None:
    """This process's GPU memory, or None where nvidia-smi cannot tell (WSL, containers, [N/A])."""
    query = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, check=True).stdout
        used = dict(line.split(", ") for line in out.strip().splitlines())
        return int(used[str(os.getpid())])
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError):
        return None


def report(args: argparse.Namespace) -> None:
    clips = _load_manifest()
    bench = _bench_fingerprint()
    accuracy_runs = [_read_json(path) for path in sorted(RESULTS_DIR.glob("accuracy__*.json"))]
    speed_runs = [_read_json(path) for path in sorted(RESULTS_DIR.glob("speed__*.json"))]
    stale = [f"{run['name']} ({run['device']})" for run in accuracy_runs + speed_runs if run.get("bench") != bench]
    current = [run for run in accuracy_runs if run.get("bench") == bench]
    speed_runs = [run for run in speed_runs if run.get("bench") == bench]
    for run in current:  # re-score the saved transcripts, so every run is scored with the same normaliser
        by_id = {row["id"]: row for row in run["clips"]}
        run["summary"] = _score(clips, [by_id[clip["id"]] for clip in clips])
    lines = [
        "# Bake-off results",
        "",
        f"## Accuracy: {CLIPS_PER_SOURCE} LinTO + {CLIPS_PER_SOURCE} FARUK test clips, greedy, normalised",
        "",
        "| Model | Device | Threads | WER all | WER LinTO | WER FARUK | CER all | RTF | Empty or `!!!` |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in sorted(current, key=lambda run: run["summary"]["all"]["wer"]):
        scores = run["summary"]
        lines.append(
            f"| {run['name']} | {run['device']} {run['compute_type']} | {run['threads']} | {scores['all']['wer']:.1f} "
            f"| {scores['linto']['wer']:.1f} | {scores['faruk']['wer']:.1f} | {scores['all']['cer']:.1f} "
            f"| {scores['all']['rtf']:.3f} | {scores['all']['degenerate']} |"
        )
    if stale:
        lines += ["", f"Left out because they ran on another bench set: {', '.join(stale)}."]
    lines += [
        "",
        f"## Speed: {SPEED_CLIP_SECONDS:.0f} s utterance, median of repeats",
        "",
        f"The final line appears min_silence ({StreamConfig().min_silence_s} s) plus about {PIPELINE_LAG_S:.2f} s "
        "of chunk and frame lag after the pause, plus the final decode; that sum is what must stay within "
        f"{LATENCY_BUDGET_S} s.",
        "",
        "| Model | Device | Threads | Partial, greedy (s) | Final, beam 3 (s) | Final line after pause (s) "
        f"| RAM (MB) | VRAM (MB) | ≤ {LATENCY_BUDGET_S} s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in speed_runs:
        after_pause = run["final_s"]["median"] + StreamConfig().min_silence_s + PIPELINE_LAG_S
        lines.append(
            f"| {run['name']} | {run['device']} {run['compute_type']} | {run['threads']} "
            f"| {run['partial_s']['median']:.2f} | {run['final_s']['median']:.2f} | {after_pause:.2f} "
            f"| {run['rss_mb']} | {run['vram_mb'] if run['vram_mb'] is not None else 'n/a'} "
            f"| {'pass' if after_pause <= LATENCY_BUDGET_S else 'FAIL'} |"
        )
    lines += ["", "## Ear test: sound.wav", ""]
    lines += [f"- **{run['name']}** ({run['device']}): {run['ear_test']['sound.wav']}" for run in current]
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


def _load(args: argparse.Namespace):
    args.compute_type = args.compute_type or ("int8_float32" if args.device == "cuda" else "int8")
    return load_model(args.model, args.device, args.compute_type, args.threads)


def _run_info(args: argparse.Namespace) -> dict:
    return {
        "name": args.name,
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "threads": args.threads,
        "bench": _bench_fingerprint(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _bench_fingerprint() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:12]


def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit("No bench set yet: run `python bakeoff.py prepare` first.")
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]


def _load_clip(entry: dict) -> np.ndarray:
    return decode_audio(str(DATA_DIR / entry["audio"]), sampling_rate=SR)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(kind: str, args: argparse.Namespace, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{kind}__{args.name}__{args.device}-{args.compute_type}-{args.threads}t.json"
    (RESULTS_DIR / name).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def _run_name(value: str) -> str:
    if not re.fullmatch(r"[\w.-]+", value):
        raise argparse.ArgumentTypeError("use only letters, digits, '.', '_' and '-'")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="sample the bench set")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--force", action="store_true", help="replace an existing bench set")
    for command in ("sanity", "accuracy", "speed"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", required=True, help="CTranslate2 directory or Hugging Face repo id")
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--compute-type", help="default: int8 on CPU, int8_float32 on GPU")
        sub.add_argument("--threads", type=int, default=4, help="CPU threads")
        if command == "sanity":
            sub.add_argument("--expect", default="يا شبينا")
        else:
            sub.add_argument("--name", required=True, type=_run_name, help="label used in the results and the report")
        if command == "speed":
            sub.add_argument("--repeats", type=int, default=5)
    commands.add_parser("report")
    args = parser.parse_args()
    {"prepare": prepare, "sanity": sanity, "accuracy": accuracy, "speed": speed, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
