import transformers


class PegasusGPT2Config(transformers.GPT2Config):
    model_type = "pegasusgpt2"

    def __init__(
        self,
        vocab_size=50000,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


transformers.AutoConfig.register(PegasusGPT2Config.model_type, PegasusGPT2Config)
