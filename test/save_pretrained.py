from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from pegasuslm import (
    PegasusGPT2Config,
    PegasusGPT2Tokenizer,
    PegasusGPT2Model
)

model_path = Path(__file__).resolve().parent.parent / "model/pegasus-gpt2-small"
# model_path = Path(__file__).resolve().parent.parent / 'model/rinna-dev'

config = PegasusGPT2Config()
config.save_pretrained(model_path)
tokenizer = PegasusGPT2Tokenizer(
    "model/50KV_200MS/spiece.model",
    model_max_length=config.n_positions,
)
tokenizer.save_pretrained(model_path)
model = PegasusGPT2Model(config)
model.save_pretrained(model_path)
