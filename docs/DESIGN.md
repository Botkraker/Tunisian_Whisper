# Design notes

## Goal
Live microphone transcription of Tunisian Derja (with FR/EN code-switching). It must run on any PC, including CPU-only ones. It must be fast, with partial text in under 1 s and a final line in under 1.5 s after the speaker pauses. Accuracy matters as much as speed. Arabic-script output is fine, and translation to EN/FR comes later as a separate step.

## Core decision: a two-tier design
One model cannot be both accurate and able to run on any PC, so the app picks a tier from the hardware.

| Tier | When | Partials (live text) | Finals (after a pause) |
|---|---|---|---|
| **Lite** (default) | any PC, CPU only | small Derja model, CT2 int8, CPU | same model, beam 3 |
| **Pro** | NVIDIA GPU with ≥4 GB | small model, int8 | large-v3 Derja model, CT2 int8_float32 (about 1.6 GB) |

The two tiers share one runtime, built from three parts:
- **faster-whisper (CTranslate2)** runs inference. It works on CPU and GPU, int8 is fast on both, and it avoids the fp16 slow path on the GTX 1650.
- **Silero VAD** finds speech. The app decodes only speech turns, so it never wastes compute on silence. This also limits the 30 s padding cost to one decode per turn.
- **The LocalAgreement-2 policy** (from `ufal/whisper_streaming`) handles live text. A word is shown as confirmed only when two decodes in a row agree on it. This stops the flicker of rewritten text.

## Candidate models (from the desk review)
- **Lite candidate:** `TheBiggyBang/whisper-small-tunisian-derja` and its `-ct2` version. It has an Apache-2.0 licence, a 0.25 GB CT2 file, a LinTO WER of 30, and a bundled `live_asr.py`. It is the front-runner.
- **Pro candidate:** `oddadmix/Whisperv3-tunisian-codeswitch`, with a NADI-2026 WER of 15.2 and code-switch training. Its licence is not stated, so ask the author before you publish anything with it.
- **Backup Pro:** `montachouig/whisper-large-v3-tunisian`, a LoRA adapter with an Apache-2.0 licence. Merge it into the base, then convert it to CT2.
- **Drop:** TuniSpeech large-v2 (CC-BY-NC, slow), ghayth123 (mislabelled, no score), and awaxsama tiny (too inaccurate). Test medfadiabaidi only if time allows.

## Steps
1. **Shorter bake-off.** Use 100 clips each from LinTO test and the FARUK test shard, plus `sound.wav` as an ear test. Use the same decoding for every model (`language="ar"`, greedy). Report normalised WER, CER, and RTF. Test only these 4 models: TheBiggyBang, medfadiabaidi, oddadmix, montachouig. Run everything through faster-whisper CT2 int8 so the scores match what ships.
2. **Speed check on the target hardware.** Run each model on CPU (4 threads) and on the GPU. Record the 5 s utterance final latency, the partial latency, and RAM/VRAM use. Pass if Lite finals take ≤1.5 s on CPU and Pro finals take ≤1.5 s on the GPU.
3. **Pick the models.** Lite gets the best small model that passes on CPU. Pro gets the best large model that passes on the GPU. If no large model passes, ship Lite only.
4. **Build `live.py`.** It takes mic or file input, runs VAD into a turn buffer, decodes greedy partials every 0.5 s with LocalAgreement, and decodes the final with beam 3 at the end of each turn. It detects the hardware and picks the tier automatically. A `--tier lite|pro` flag overrides the choice. Base it on the `TurnTranscriber` in `live_asr.py`.
5. **Package the app** with `pip install faster-whisper sounddevice`. On Linux, also run `apt install libportaudio2`. The CT2 models download the first time the app runs.
6. **Later:** translate Derja to EN/FR, with an NLLB or MarianMT model run on each final line.

## Verification
- Run the harness sanity check: TuniSpeech on `sound.wav` chunk 0 must give 'يا شبينا', and normalisation must treat أ, إ, and ة variants as matches.
- Reject any output that is empty or `!!!!` (the NaN symptom).
- Replay `sound.wav` through `live.py --file`, then do a real mic test on both tiers.
- Check on a second, CPU-only PC, because "any PC" is the requirement.

## Machine limits (dev PC)
The dev PC has a GTX 1650 with 4 GB of VRAM, about 1 GB of free RAM, and about 16 GB of free disk. Delete a model's HF cache after its bake-off run unless it is a finalist.

## Licence notes
TheBiggyBang and montachouig are Apache-2.0, so they are safe to use. oddadmix has no licence stated, so ask the author first. TuniSpeech is non-commercial (CC-BY-NC).
