from transformers import PretrainedConfig


class NanoGPTConfig(PretrainedConfig):
    model_type = 'nanogpt'

    def __init__(
        self,
        vocab_size=5000,
        block_size=513,
        n_layer=1,
        n_head=1,
        n_embd=384,
        bias=False,
        is_causal=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.hidden_size = n_embd  # read by the BabyLM finetuning classifier
        self.bias = bias
        self.is_causal = is_causal
        kwargs['tie_word_embeddings'] = False  # lm_head reuses wte directly
        super().__init__(**kwargs)
