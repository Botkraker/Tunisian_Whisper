# Bake-off results

## Accuracy: 100 LinTO + 100 FARUK test clips, greedy, normalised

| Model | Device | Threads | WER all | WER LinTO | WER FARUK | CER all | RTF | Empty or `!!!` |
|---|---|---|---|---|---|---|---|---|
| montachouig-large | cuda int8_float32 | 4 | 26.6 | 13.1 | 36.0 | 13.3 | 0.256 | 0 |
| oddadmix-large | cuda int8_float32 | 4 | 30.5 | 28.6 | 31.8 | 10.8 | 0.250 | 0 |
| tbb-small | cuda int8_float32 | 4 | 41.1 | 31.0 | 48.2 | 20.6 | 0.061 | 0 |
| tbb-small | cpu int8 | 4 | 42.5 | 32.6 | 49.4 | 20.8 | 0.616 | 0 |
| medfadiabaidi-small | cpu int8 | 4 | 50.2 | 48.8 | 51.1 | 23.7 | 0.537 | 0 |
| medfadiabaidi-small | cuda int8_float32 | 4 | 51.6 | 47.5 | 54.5 | 24.5 | 0.059 | 0 |

## Speed: 5 s utterance, median of repeats

The final line appears min_silence (0.5 s) plus about 0.13 s of chunk and frame lag after the pause, plus the final decode; that sum is what must stay within 1.5 s.

| Model | Device | Threads | Partial, greedy (s) | Final, beam 3 (s) | Final line after pause (s) | RAM (MB) | VRAM (MB) | ≤ 1.5 s |
|---|---|---|---|---|---|---|---|---|
| montachouig-large | cpu int8 | 4 | 10.40 | 10.56 | 11.19 | 2180 | n/a | FAIL |
| montachouig-large | cuda int8_float32 | 4 | 1.26 | 1.39 | 2.03 | 620 | 2016 | FAIL |
| oddadmix-large | cpu int8 | 4 | 10.60 | 10.98 | 11.61 | 2217 | n/a | FAIL |
| oddadmix-large | cuda int8_float32 | 4 | 1.26 | 1.45 | 2.08 | 621 | 2016 | FAIL |
| tbb-small | cpu int8 | 12 | 3.58 | 4.92 | 5.55 | 518 | n/a | FAIL |
| tbb-small | cpu int8 | 4 | 2.14 | 2.19 | 2.82 | 571 | n/a | FAIL |
| tbb-small | cpu int8 | 6 | 1.89 | 1.94 | 2.57 | 562 | n/a | FAIL |
| tbb-small | cuda int8_float32 | 4 | 0.30 | 0.31 | 0.94 | 505 | 416 | pass |

## Ear test: sound.wav

- **medfadiabaidi-small** (cpu): يا شبينا اللال يا شبينا اللال خمسة ولخ ميتسانين آ اللال آي صحي الشبي هذا تشبي هذا   باهي ما حاجتي بيتش أوقف أوقف يمشي وخ قدم ووقف حبس حبس
- **medfadiabaidi-small** (cuda): يا شبينا اللالة يا شبينا اللالة خمسة ولق ميتسانين آ اللالة أي صحي الشبي هذا تشبي هذا   باهي حاجة تبيكش وق وق يمشي وخ قدم وق حبس حبس
- **montachouig-large** (cuda): يا شبينا ألالا يا شبينا ألالا خمسة ولا خمسانين ألالا أجزحا شبيه هاذا تي شبيه هاذا كلمو باهي حاشتي بيكش وقف وقف إمشي وخر قدم وقف حبس حبس
- **oddadmix-large** (cuda): يا شبينا للا يا شبينا للا خمسة ولا خميسة نينا للا يا زحايا شبيه هذا تي شبيه هذا كلمو باهي حاشتي بيك شوف اوقف اوقف امشي وخر قدم اوقف حبس حبس
- **tbb-small** (cpu): يا شبينا لا لا يا شبينا لا لا خمسة و الاخمي تانينا لا لا أيه يا سعي شبي هذا تشبي هذا تتكلموا باهي محاجة تبيكشي اوقف اوقف يمشي واخر قدم اوقف حبس حبس
- **tbb-small** (cuda): يا شبينا لا لا يا شبينا لا لا خمسة و الاخمي تانينا لا لا أيه يا زيحة شبي هذا تشبي هذا تتكلموا باهي محشة بكشية وقفوقف نمشي واخر قدموقف حبس حبس
