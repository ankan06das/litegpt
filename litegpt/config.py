from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    p_drop: float = 0.1
    d_ff: int = 2048
    max_len: int = 512