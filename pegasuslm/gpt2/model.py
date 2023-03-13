import transformers
from .config import PegasusGPT2Config


class PegasusGPT2Model(transformers.GPT2LMHeadModel):
    config_class = PegasusGPT2Config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


transformers.AutoModelForCausalLM.register(PegasusGPT2Config, PegasusGPT2Model)
