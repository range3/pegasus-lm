# range3/pegasus-lm

## setup
```bash
module load cuda/11.8.0
module load cudnn/8.6.0/cuda11
module load openmpi/4.1.4/gcc9.4.0-cuda11.8.0
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
pip install neologdn \
  prefetch-generator \
  datasets \
  sentencepiece \
  transformers \
  scikit-learn \
  evaluate \
  tensorboard \
  accelerate \
  git+https://github.com/huggingface/transformers
```
