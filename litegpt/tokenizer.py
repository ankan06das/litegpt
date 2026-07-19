import tiktoken

class GPT2Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.base_vocab_size = self.enc.n_vocab
    
        self.special_tokens = {
            "<PAD>": self.base_vocab_size, # 50256
            "<BOS>": self.base_vocab_size + 1, # 50257
            "<EOS>": self.base_vocab_size + 2, # 50258
        }
        self.vocab_size = self.base_vocab_size + 3
    
    @property
    def pad_id(self):
        return self.special_tokens['<PAD>']
    
    @property
    def bos_id(self):
        return self.special_tokens['<BOS>']
    
    @property
    def eos_id(self):
        return self.special_tokens['<EOS>']
    
    def encode(self, text:str, add_special_tokens: bool = True) -> list[int]:
        ids = self.enc.encode(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in self.special_tokens.values()]
        return self.enc.decode(ids)