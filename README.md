# Tunisian Whisper

Live transcription of Tunisian Derja (Tunisian Arabic, with the usual French and English mixed in) from your microphone, or from an audio file replayed in real time. It runs on faster-whisper / CTranslate2 in int8, uses Silero VAD to find speech turns, and shows live partial text with a LocalAgreement-2 policy so words don't keep flickering. It picks a tier from your hardware: a small Derja model on the CPU (Lite) or, if you have an NVIDIA GPU, a large-v3 Derja model for the final lines (Pro). I also wrote a small bake-off harness to choose the models; its results are below.

## Features

- Microphone input, or `--file` to replay a recording at real-time speed
- Two tiers picked automatically from the hardware, with `--tier` to override
  - **Lite**: CPU only, `TheBiggyBang/whisper-small-tunisian-derja-ct2` for both partials and finals (downloads on first run)
  - **Pro**: NVIDIA GPU with at least 4 GB, small model for partials, a large-v3 Derja model (`models/oddadmix-ct2`) for finals
- If Pro fails to load in auto mode, it falls back to Lite
- Greedy partials while you speak, beam-3 final after each pause
- Arabic text is reshaped and run through bidi so it reads correctly in a terminal
- Bake-off harness (`bakeoff.py`) for accuracy (WER/CER/RTF) and latency of any CTranslate2 Whisper model

## Requirements

- Python 3.10+
- On Linux, PortAudio for the microphone:
  ```bash
  sudo apt install libportaudio2
  ```
- Pro tier only: an NVIDIA GPU with 4 GB or more, plus the CUDA libraries from pip:
  ```bash
  pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  ```

## Install

```bash
git clone https://github.com/Botkraker/tunisian-whisper.git
cd tunisian-whisper
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Usage

```bash
python live.py                       # microphone, tier picked from the hardware
python live.py --file recording.wav  # replay a file in real time
python live.py --tier lite           # force the CPU tier
python live.py --tier pro            # force the GPU tier (needs models/oddadmix-ct2)
python live.py --threads 6           # CPU threads (default: min(4, CPU count))
```

`--tier` takes `auto` (default), `lite` or `pro`. The tier, device and final model in use are printed to stderr at startup.

## Getting the Pro model

The Pro final model isn't downloaded automatically. Convert it once into `models/oddadmix-ct2`:

```bash
pip install -r requirements-dev.txt
python convert_models.py oddadmix
```

Add `--delete-source` to remove the downloaded Hugging Face checkpoint afterwards. The other names `convert_models.py` accepts are `medfadiabaidi`, `montachouig` (merges the LoRA adapter into `openai/whisper-large-v3` first) and `tunispeech`; those are only needed for the bake-off. Without `models/oddadmix-ct2`, auto mode always picks Lite.

## Bake-off

`bakeoff.py` scores models with the same decoding for all of them (greedy, `language="ar"`, temperature 0, no timestamps; int8 on CPU, int8_float32 on GPU). It needs the dev requirements.

```bash
pip install -r requirements-dev.txt

python bakeoff.py prepare                                   # sample 100 LinTO + 100 FARUK test clips
python bakeoff.py sanity   --model models/tunispeech-ct2    # sound.wav chunk 0 must read "يا شبينا"
python bakeoff.py accuracy --model models/oddadmix-ct2 --name oddadmix-large --device cuda
python bakeoff.py speed    --model TheBiggyBang/whisper-small-tunisian-derja-ct2 --name tbb-small --device cpu --threads 4
python bakeoff.py report                                    # writes bench/results/REPORT.md
```

`--model` takes a CTranslate2 directory or a Hugging Face repo id. Other options: `prepare --seed N --force`, `--compute-type`, `speed --repeats N`, `sanity --expect TEXT`.

The bench clips, the converted models and `sound.wav` (the ear-test recording) are not committed. `prepare` downloads and samples the clips again from `linagora/linto-dataset-audio-ar-tn` and `FARUKxAUTO/tunisian-asr-cleaned`, and `convert_models.py` rebuilds the models. `bench/data/manifest.jsonl` and the JSON results in `bench/results/` are kept so the report can be regenerated.

### Results

Accuracy on 100 LinTO + 100 FARUK test clips (greedy, normalised, %), trimmed from [bench/results/REPORT.md](bench/results/REPORT.md):

| Model | Device | WER all | WER LinTO | WER FARUK | CER all | RTF |
|---|---|---|---|---|---|---|
| montachouig-large | cuda int8_float32 | 26.6 | 13.1 | 36.0 | 13.3 | 0.256 |
| oddadmix-large | cuda int8_float32 | 30.5 | 28.6 | 31.8 | 10.8 | 0.250 |
| tbb-small | cpu int8 | 42.5 | 32.6 | 49.4 | 20.8 | 0.616 |
| medfadiabaidi-small | cpu int8 | 50.2 | 48.8 | 51.1 | 23.7 | 0.537 |

Latency for a 5 s utterance (final line after the pause, target 1.5 s), on my dev PC with a GTX 1650:

| Model | Device | Threads | Partial (s) | Final line after pause (s) | VRAM (MB) |
|---|---|---|---|---|---|
| tbb-small | cpu int8 | 4 | 2.14 | 2.82 | n/a |
| tbb-small | cpu int8 | 6 | 1.89 | 2.57 | n/a |
| tbb-small | cuda int8_float32 | 4 | 0.30 | 0.94 | 416 |
| oddadmix-large | cuda int8_float32 | 4 | 1.26 | 2.08 | 2016 |
| montachouig-large | cuda int8_float32 | 4 | 1.26 | 2.03 | 2016 |

Pro uses oddadmix for finals; the full tables and the ear test are in the report.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
Tunisian_Whisper/
├── live.py               # the app: audio input, VAD, LocalAgreement-2, tiers
├── derja_text.py         # Arabic normalisation and WER/CER scoring
├── bakeoff.py            # bake-off harness (prepare/sanity/accuracy/speed/report)
├── convert_models.py     # HF checkpoints -> CTranslate2 in models/
├── requirements.txt      # runtime deps
├── requirements-dev.txt  # bake-off, conversion and tests
├── docs/DESIGN.md        # design notes and model choices
├── bench/
│   ├── data/             # manifest.jsonl, snapshot.json (clips/ is not committed)
│   └── results/          # per-run JSON and REPORT.md
├── experiments/          # notebook and a plain Hugging Face baseline
├── models/               # converted models (not committed)
└── tests/
```

## Known limitations

- Lite finals on CPU take about 2.6 s after the pause (6 threads) against the 1.5 s target I set. It's usable, but not as snappy as I wanted. More threads don't help: 12 threads was slower than 6.
- The large models are far too slow on CPU (over 11 s per final), so Pro really needs a GPU, and even there the finals land at about 2 s.
- GTX 16xx cards (tested on a GTX 1650) return NaN / `!!!` output from cuDNN fp16 convolutions. That's why GPU runs use `int8_float32` instead of `int8_float16` or `float16`.
- Output is Arabic script only. Translation to English/French is planned but not done.

## Model licences

- `TheBiggyBang/whisper-small-tunisian-derja(-ct2)`: Apache-2.0
- `montachouig/whisper-large-v3-tunisian`: Apache-2.0
- `oddadmix/Whisperv3-tunisian-codeswitch`: no licence stated. Check with the author before redistributing it or anything built on it.
- `TuniSpeech-AI/whisper-tunisian-dialect`: CC-BY-NC, used only for the harness sanity check.

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and CTranslate2
- [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) for the LocalAgreement policy
- [Silero VAD](https://github.com/snakers4/silero-vad)
- The LinTO (`linagora/linto-dataset-audio-ar-tn`) and FARUK (`FARUKxAUTO/tunisian-asr-cleaned`) datasets
- The model authors: TheBiggyBang, oddadmix, montachouig, medfadiabaidi and TuniSpeech-AI

## License

No licence has been chosen yet.

<!-- TODO: add the licence here -->
