#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${BASE_MODEL:-google/gemma-3-270m-it}"
MERGED_DIR="output/merged"
GGUF_DIR="output/gguf"
LLAMA_CPP_DIR="output/llama.cpp"
mkdir -p "$MERGED_DIR" "$GGUF_DIR"

python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = os.environ.get("BASE_MODEL", "google/gemma-3-270m-it")
token = os.environ["HF_TOKEN"]
model = AutoModelForCausalLM.from_pretrained(base, token=token, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, "output/adapter")
model.merge_and_unload().save_pretrained("output/merged", safe_serialization=True, max_shard_size="1GB")
AutoTokenizer.from_pretrained(base, token=token).save_pretrained("output/merged")
PY

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_NATIVE=OFF -DGGML_OPENMP=ON
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j2
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$GGUF_DIR/gemma-3-270m-indonesian-finetune-f16.gguf" --outtype f16
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_DIR/gemma-3-270m-indonesian-finetune-f16.gguf" "$GGUF_DIR/gemma-3-270m-indonesian-finetune-Q8_0.gguf" Q8_0
rm -f "$GGUF_DIR/gemma-3-270m-indonesian-finetune-f16.gguf"
ls -lh "$GGUF_DIR"
