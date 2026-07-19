from typing import Union

import math

import torch
import torch.nn as nn
from .config import GPTConfig

class EmbeddingLayer(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        self.tk_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.p_drop)
        
    def forward(self, x, pos):
        return self.dropout(self.tk_embed(x) + self.pos_embed(pos))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        
        self.cfg = cfg
        self.d_head = self.cfg.d_model // self.cfg.num_heads
        
        self.W = nn.Linear(self.cfg.d_model, self.cfg.d_model * 3, bias=False)
        
        self.Wo = nn.Linear(self.cfg.d_model, self.cfg.d_model, bias=False)
        self.Wo.SCALE_INIT = 1
        
        self.dropout = nn.Dropout(self.cfg.p_drop)
        
    def _attention(self, Q, K, V, mask=None):
        
        scores = Q@K.transpose(-2, -1)/math.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.dropout(weights)@V, weights
    
    def forward(self,x:torch.Tensor, mask=None):
        B, L, _ = x.shape
        
        Q,K,V = self.W(x).chunk(3, dim=-1)
        Q = Q.view(B, L, self.cfg.num_heads, self.d_head).transpose(1,2)
        K = K.view(B, L, self.cfg.num_heads, self.d_head).transpose(1,2)
        V = V.view(B, L, self.cfg.num_heads, self.d_head).transpose(1,2)
        
        attn, last_attn = self._attention(Q, K, V, mask)
        self.last_attn = last_attn.detach()
        O = attn.reshape(B, L, self.cfg.d_model)
        return self.Wo(O)

class FFN(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.p_drop),
        )
        self.final = nn.Linear(cfg.d_ff,cfg.d_model)
        self.final.SCALE_INIT = 1
        
    def forward(self, x):
        return self.final(self.ffn(x))
 
class PreLN(nn.Module):
    def __init__(self, SubLayer: Union[MultiHeadAttention, FFN], cfg: GPTConfig):
        super().__init__()
        self.sl = SubLayer
        self.dropout = nn.Dropout(cfg.p_drop)
        self.layernorm = nn.LayerNorm(cfg.d_model)
        
    def forward(self, x, *args, **kwargs):
        return x + self.dropout(self.sl(self.layernorm(x), *args, **kwargs))
class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.masked_attn_layer = PreLN(MultiHeadAttention(self.cfg), self.cfg)
        self.ffn_layer = PreLN(FFN(self.cfg), self.cfg)
    
    def forward(self, x, mask=None):
        x = self.masked_attn_layer(x, mask=mask)
        return self.ffn_layer(x)
    

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embedding = EmbeddingLayer(cfg)
        self.blocks = nn.ModuleList([
            GPTBlock(self.cfg)
            for _ in range(self.cfg.num_layers)
        ])
        self.final_ln = nn.LayerNorm(self.cfg.d_model)  
        self.out_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.cfg.num_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.00, std=0.02)
            
            
    def forward(self, x, mask=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = self.embedding(x, pos)
        
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.final_ln(x)
        logits = self.out_head(x)
        return logits
        
        
    
        