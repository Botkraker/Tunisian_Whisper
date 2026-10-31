"""Convert the Hugging Face bake-off candidates to CTranslate2 int8.

    python convert_models.py medfadiabaidi oddadmix montachouig tunispeech [--delete-source]

Writes models/<name>-ct2. Checkpoints load as float16 to halve RAM use, but a large-v3 conversion still
needs about 5 GB of free RAM, so close other memory-hungry apps first. --delete-source removes the downloaded
Hugging Face checkpoint once its CT2 copy exists.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from ctranslate2.converters import TransformersConverter
from huggingface_hub import file_exists, hf_hub_download

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class Candidate:
    repo: str
    adapter: str | None = None
    copy_files: tuple[str, ...] = ("tokenizer.json", "preprocessor_config.json")  # faster-whisper reads mel bins here
    min_free_gb: float = 5.0
    keep_source: bool = False


CANDIDATES = {
    "medfadiabaidi": Candidate("medfadiabaidi/whisper-small-tunisian-asr", min_free_gb=2.0),
    "oddadmix": Candidate("oddadmix/Whisperv3-tunisian-codeswitch"),
    "montachouig": Candidate("openai/whisper-large-v3", adapter="montachouig/whisper-large-v3-tunisian"),
    # Only for the harness sanity check. It is a large-v2 (80 mel bins, faster-whisper's default) without a
    # preprocessor_config.json, and experiments/whisper_hf_baseline.py uses its cache, so the cache is never deleted.
    "tunispeech": Candidate("TuniSpeech-AI/whisper-tunisian-dialect", copy_files=("tokenizer.json",), keep_source=True),
}


class LoraMergingConverter(TransformersConverter):
    """Loads the base checkpoint and folds a LoRA adapter into its weights before converting."""

    def __init__(self, base: str, adapter: str, **kwargs):
        super().__init__(base, **kwargs)
        self._base = base
        self._adapter = adapter

    def load_model(self, model_class, model_name_or_path, **kwargs):
        from safetensors.torch import load_file

        model = super().load_model(model_class, model_name_or_path, **kwargs)
        config = json.loads(Path(hf_hub_download(self._adapter, "adapter_config.json")).read_text())
        weights = load_file(hf_hub_download(self._adapter, "adapter_model.safetensors"))
        merge_lora(model, config, weights, base=self._base)
        return model


def merge_lora(model: torch.nn.Module, config: dict, weights: dict[str, torch.Tensor], base: str) -> None:
    """Folds a plain LoRA adapter into model: W += (alpha / r) * B @ A for every adapted layer."""
    if config.get("peft_type") != "LORA":
        raise ValueError(f"not a LoRA adapter (peft_type {config.get('peft_type')})")
    options = ("use_dora", "use_rslora", "use_qalora", "rank_pattern", "alpha_pattern", "lora_bias", "layer_replication", "target_parameters")
    unsupported = [key for key in options if config.get(key)]
    if unsupported or config.get("bias", "none") != "none":
        raise ValueError(f"not a plain LoRA adapter: {unsupported or ['bias=' + str(config['bias'])]}")
    if config.get("base_model_name_or_path") != base:
        raise ValueError(f"the adapter was trained on {config.get('base_model_name_or_path')}, not on {base}")
    prefixes = {key.removesuffix(".lora_A.weight") for key in weights if key.endswith(".lora_A.weight")}
    expected = {f"{prefix}.lora_{side}.weight" for prefix in prefixes for side in "AB"}
    if not prefixes or set(weights) != expected:
        raise ValueError(f"the adapter is not plain lora_A/lora_B pairs: {sorted(set(weights) ^ expected)[:3]}")
    scale = config["lora_alpha"] / config["r"]
    modules = dict(model.named_modules())
    with torch.no_grad():
        for prefix in sorted(prefixes):
            name = prefix.removeprefix("base_model.model.")
            if name not in modules:
                raise ValueError(f"the model has no layer {name}")
            target = modules[name]
            delta = scale * weights[f"{prefix}.lora_B.weight"].float() @ weights[f"{prefix}.lora_A.weight"].float()
            if delta.shape != target.weight.shape:
                raise ValueError(f"{name}: a {tuple(delta.shape)} update does not fit its {tuple(target.weight.shape)} weight")
            target.weight.copy_((target.weight.float() + delta).to(target.weight.dtype))
    print(f"merged {len(prefixes)} LoRA pairs (scale {scale:g})")


def convert(name: str, delete_source: bool) -> bool:
    candidate = CANDIDATES[name]
    output_dir = MODELS_DIR / f"{name}-ct2"
    if output_dir.is_dir():
        print(f"{name}: {output_dir} already exists, skipping")
        return True
    try:
        missing = [filename for filename in candidate.copy_files if not file_exists(candidate.repo, filename)]
    except Exception:  # offline: converting from the cache still works, and the converter names any missing file
        missing = []
    if missing:
        print(f"{name}: {candidate.repo} has no {', '.join(missing)}", file=sys.stderr)
        return False
    free_gb = psutil.virtual_memory().available / 2**30
    if free_gb < candidate.min_free_gb:
        print(
            f"{name}: needs about {candidate.min_free_gb:.0f} GB of free RAM but only {free_gb:.1f} GB is available. "
            "Close other apps, then retry.",
            file=sys.stderr,
        )
        return False
    options = {"copy_files": list(candidate.copy_files), "load_as_float16": True}
    if candidate.adapter:
        converter = LoraMergingConverter(candidate.repo, candidate.adapter, **options)
    else:
        converter = TransformersConverter(candidate.repo, **options)
    partial_dir = output_dir.with_name(output_dir.name + ".partial")  # renamed only once the conversion is complete
    converter.convert(str(partial_dir), quantization="int8_float16", force=True)
    os.replace(partial_dir, output_dir)
    print(f"{name}: wrote {output_dir}")
    if delete_source and not candidate.keep_source:
        for repo in filter(None, (candidate.repo, candidate.adapter)):
            _delete_cached_repo(repo)
    return True


def _delete_cached_repo(repo_id: str) -> None:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            strategy = cache.delete_revisions(*(revision.commit_hash for revision in repo.revisions))
            print(f"deleting the {repo_id} download ({strategy.expected_freed_size_str})")
            strategy.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", choices=sorted(CANDIDATES))
    parser.add_argument("--delete-source", action="store_true", help="delete the downloaded checkpoint afterwards")
    args = parser.parse_args()
    converted = []
    for name in args.names:
        try:
            converted.append(convert(name, args.delete_source))
        except Exception as error:  # one candidate failing (bad adapter, full disk, network) must not stop the rest
            print(f"{name}: conversion failed: {error}", file=sys.stderr)
            converted.append(False)
    sys.exit(0 if all(converted) else 1)


if __name__ == "__main__":
    main()
