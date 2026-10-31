import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from convert_models import merge_lora

PREFIX = "base_model.model.model.encoder.layers.0.self_attn.q_proj"
CONFIG = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none", "base_model_name_or_path": "openai/whisper-large-v3"}


def tiny_whisper() -> WhisperForConditionalGeneration:
    torch.manual_seed(0)
    config = WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
        num_mel_bins=8,
        max_source_positions=16,
        max_target_positions=16,
    )
    return WhisperForConditionalGeneration(config)


def adapter() -> dict[str, torch.Tensor]:
    return {f"{PREFIX}.lora_A.weight": torch.ones(2, 8), f"{PREFIX}.lora_B.weight": torch.full((8, 2), 0.5)}


def test_merge_adds_the_scaled_low_rank_update():
    model = tiny_whisper()
    before = model.model.encoder.layers[0].self_attn.q_proj.weight.detach().clone()
    merge_lora(model, CONFIG, adapter(), base="openai/whisper-large-v3")
    # B @ A is 1.0 everywhere and alpha / r is 2, so every weight grows by 2.0
    torch.testing.assert_close(model.model.encoder.layers[0].self_attn.q_proj.weight, before + 2.0)


@pytest.mark.parametrize(
    ("config", "weights"),
    [
        ({**CONFIG, "lora_bias": True}, adapter()),  # its lora_B.bias would be silently dropped
        (CONFIG, {**adapter(), f"{PREFIX}.lora_B.bias": torch.zeros(8)}),  # a bias the config does not declare
        ({**CONFIG, "peft_type": "IA3"}, adapter()),  # not LoRA at all
        ({**CONFIG, "layer_replication": [[0, 1]]}, adapter()),  # changes the architecture
        ({**CONFIG, "base_model_name_or_path": "openai/whisper-large-v2"}, adapter()),  # trained on another base
        (CONFIG, {f"{PREFIX}.lora_A.weight": torch.ones(2, 8)}),  # lora_A without its lora_B
        (CONFIG, {key.replace("layers.0", "layers.9"): value for key, value in adapter().items()}),  # no such layer
        (CONFIG, {}),  # nothing to merge
    ],
)
def test_merge_refuses_adapters_it_cannot_apply_faithfully(config, weights):
    with pytest.raises(ValueError):
        merge_lora(tiny_whisper(), config, weights, base="openai/whisper-large-v3")
