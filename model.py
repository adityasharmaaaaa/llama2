import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4095
    n_layers: int = 32
    n_heads: int= 32 #number of heads for the queries 
    n_kv_heads = Optional[int] = None #number of heads for k and v
    vocab_size: int = -1 #this will be set when we load the tokenizer
    multiple_of=256
    ffn_dim_multiplier: Optional[float] = None
    norm_e = 1e-5

    #needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_thetha_pos_frequencies(head_dim: int, seq_len:int, device: str, thetha: float = 10000.0):
    #as written in the paper, the dimension of the embedding must be even
    assert head_dim%2==0, "Dimension must be even"
    #build the theta parameters
    #according to the formula theta_i=10000^(-2(i-1)/dim) for i =[1,2,...dim/2]
    #shape: (head_dim/2)
    theta_numerator = torch.arange(0,head_dim,2).float()
    theta = 1.0/(theta ** (theta_numerator/head_dim)).to(device)
    #construct the positions (the "m parameter")
    m = torch.arange(seq_len, device=device)
    #multiply each theta by each position using the outer product
    freqs=torch.outer(m,theta).float()
    #we can compute complex numbers in the polar form c=R*exp(i*m*theta) where R=1
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor)

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs)->None:
        super().__init__()

        assert args.vocab_size != -1 #vocab size must be set

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim)

        self.layers=nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_thetha_pos_frequencies(self.args_dim // self.args.n_heads, self.args.max_seq_len*2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        #(b,seq_len)
        batch_size,seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"

        #(b,seq_len)->(b,seq_len,dim)
        h=self.tok_embeddings(tokens)

        #retrieve the pairs(m,thetha) corresponding to the positions [start_pos, start_pos+seq_len]
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]

        #consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output