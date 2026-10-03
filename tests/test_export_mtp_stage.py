import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_mtp_stage", ROOT / "scripts" / "export-mtp-stage.py"
)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def test_mtp_export_selects_quantized_shared_matrices_and_complete_mtp_layer():
    names = {
        "model.language_model.embed_tokens.weight": "a.safetensors",
        "model.language_model.embed_tokens.scales": "a.safetensors",
        "model.language_model.embed_tokens.biases": "a.safetensors",
        "lm_head.weight": "b.safetensors",
        "lm_head.scales": "b.safetensors",
        "lm_head.biases": "b.safetensors",
        "model.language_model.layers.45.eh_proj.weight": "c.safetensors",
        "model.language_model.layers.45.shared_head.norm.weight": "c.safetensors",
        "model.language_model.layers.44.input_layernorm.weight": "c.safetensors",
    }

    selected = EXPORTER.selected_names(names, 45)

    assert selected == sorted(set(names) - {"model.language_model.layers.44.input_layernorm.weight"})


def test_mtp_export_requires_both_shared_weight_matrices_and_the_mtp_layer():
    with pytest.raises(ValueError, match="embedding or language-model head"):
        EXPORTER.selected_names({
            "model.language_model.embed_tokens.weight": "a.safetensors",
            "model.language_model.layers.45.eh_proj.weight": "c.safetensors",
        }, 45)
    with pytest.raises(ValueError, match="lacks MTP layer 45"):
        EXPORTER.selected_names({
            "model.language_model.embed_tokens.weight": "a.safetensors",
            "lm_head.weight": "b.safetensors",
        }, 45)
