#!/bin/bash
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE:-$0}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"

MODEL="model/pegasus-gpt2-medium"

cd "${PROJECT_DIR}"
source .venv/bin/activate

export PYTHONPATH="${PROJECT_DIR}"
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_CACHE=/scr/cache

latest_checkpoint=$(ls -td1 "${MODEL}"/checkpoint-* 2> /dev/null | head -1)
model_name=${latest_checkpoint:-$MODEL}

python -m pegasuslm.gpt2.exe.run_generation \
  --model_name_or_path "${model_name}" \
  --repetition_penalty 1.1 \
  --temperature 0.8 \
  --length 200  \
  --num_return_sequences 3 \
  --prompt "$1"
