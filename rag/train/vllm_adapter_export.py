"""Export a PEFT adapter trained on Qwen3_5ForCausalLM for vLLM serving.

vLLM serves the full multimodal Qwen3_5ForConditionalGeneration, whose text
backbone lives under the ``language_model.`` prefix — PEFT keys saved from the
text-only class (``base_model.model.model.*``) match nothing there and are
IGNORED SILENTLY (verified 2026-08-04: identical greedy logprobs; see
CLAUDE.md dummy-LoRA round-trip entry).  This util rewrites the keys and MUST
be used for every adapter that a vLLM engine will load.

Discipline: after any change to hot-swap wiring, run a behavioral check
(base vs adapter first-token logprob diff > 1e-3) — load success proves
nothing for LoRA application.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_SRC_PREFIX = "base_model.model.model."
_DST_PREFIX = "base_model.model.language_model.model."


def export_for_vllm(adapter_dir: str | Path, out_dir: str | Path) -> Path:
    """Copy ``adapter_dir`` to ``out_dir`` with vLLM-compatible key names."""
    from safetensors.torch import load_file, save_file

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(adapter_dir / "adapter_model.safetensors")
    renamed = {}
    rewritten = 0
    for key, value in tensors.items():
        if key.startswith(_SRC_PREFIX):
            key = _DST_PREFIX + key[len(_SRC_PREFIX):]
            rewritten += 1
        renamed[key] = value
    if rewritten == 0:
        raise ValueError(
            f"no keys under {_SRC_PREFIX!r} in {adapter_dir} — wrong adapter "
            "layout, refusing to export something vLLM would silently ignore"
        )
    save_file(renamed, out_dir / "adapter_model.safetensors")
    for aux in ("adapter_config.json", "README.md"):
        if (adapter_dir / aux).exists():
            shutil.copy(adapter_dir / aux, out_dir / aux)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = export_for_vllm(args.adapter, args.out)
    print(f"exported vLLM-servable adapter -> {path}")
