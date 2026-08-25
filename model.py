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
    norm_eps= 1e-5

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

def apply_rotary_embeddings(x: torch.Tensor,freqs_complex: torch.Tensor, device:str):
    # (B,sew_len, h, head_dim)->(B,seq_len,h,head_dim/2)
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    # (seq_len, head_Dim/2)->(1,seq_len,1,head_dim/2)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B,seq_len,H,head_dim/2)*(1,seq_len,1,head_dim/2)->(B,seq_len,h,head_dim/2)
    x_rotated=x_complex*freqs_complex
    #(b,seq_len,h,head_dim/2)->(b,seq_len,h,head_dim/2,2)
    x_out=torch.view_as_real(x_rotated)
    #(b,seq_len,h,head_dim/2,2)->(B,seq_len,h,head_dim)
    x_out=x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):

    def __init__(self,dim:int,eps: float = 1e-6):
        super().__init__()
        self.eps=eps
        #gamma parameter
        self.weight=nn.Parameter(torch.ones_like(dim))

    def _norm(self,x:torch.Tensor):
        #(B,seq_lenmdim)*(B,seq_len,1)=(B,seq_lendim)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x:torch.Tensor):
        #(dim)*(B,seq_len,Dim)=(B,seq_len,dim)
        return self.weight*self._norm(x.float()).type_as(x)

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    batch_size,seq_len,n_kv_heads,head_dim=x.shape
    if n_rep==1:
        return x
    return(
        #(B,seq_len,N_KV_heads,1,head_dim)
        x[:,:,:,None,:]
        .expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim)
        .reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
    )


class SelfAttention(nn.Module):

    def __init__(self,args:ModelArgs):
        super().__init__()

        #indicates the number of heads for key and values
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        #indicates the number of heads for queries
        self.n_heads_q=args.n_heads
        #indicates how many times the heads of keys and values should be repeated to match the head of the queries
        self.n_rep=self.n_heads_q//self.n_kv_heads
        #indicates the dimension of each head
        self.head_dim=args.dim//args.n_heads

        self.wq=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        self.cache_k=torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
        self.cache_v=torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        batch_size,seq_len,_=x.shape #(B,1,dim)

        #apply the Wq, Wk and Wv matrices to queries keys and values
        #(B,1,dim) -> (B,1,H_Q*head_dim)
        xq=self.wq(x)
        #(B,1,dim)->(B,1,M_KV*head_dim)
        xk=self.wk(x)
        xv=self.wv(x)

        #(B,1,H_Q*head_dim)-->(B,1,H_Q,head_dim)
        xq=xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        xk=xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        #(B,1,H_KV*head_dim)-->(B,1J_KV,head_dim)
        xv=xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)

        xq=apply_rotary_embeddings(xq,freqs_complex,device=x.device)
        xk=apply_rotary_embeddings(xk,freqs_complex,device=x.device)
 
        #replace the entry in the cache for this token
        self.cache_k[:batch_size,start_pos:start_pos+seq_len]=xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len]=xv

        #retrieve all the cached keys and values so far
        #(B,seq_len_KV,H_KV,head_dim)
        keys=self.cache_k[:batch_size,0:start_pos+seq_len]
        values=self.cache_v[:batch_size,0:start_pos+seq_len]

        #repeat the heads of k and v to reach the number of heads of the queries
        keys=repeat_kv(keys,self.n_rep) 
        values=repeat_kv(values,self.n_rep)

        #(B,1,H_q,head_dim)-->(B,H_q,1,head_dim)
        xq=xq.trace(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(2,1)

        #(B,H_Q,1,head_dim)@(B,H_q,head_dim,seq_len_kv)-->(B,H_Q,1,seq_len_kv)
        scores=torch.matmul(xq,keys.transpose(2,3))//math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)

        #(B,H_Q,1,seq_len)@(B,H_Q,seq_len_kv,head_dim)-->(B,H_Q,1,Head_Dim)
        output=torch.matmul(scores,values)

        #(B,H_Q,1,head_dim)->(B,1,H_Q,head_dim)-->(B,1,Dim)
        output=(output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output) #(B,1,dim)->(B,1,dim)


class EncoderBlock(nn.Module):

    def __init__(self,args:ModelArgs):
        super().__init__()

        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)
        #normalisation before self attention
        self.attention_norm=RMSNorm(args.dim,eps=args.norm_eps)
        #normalisation before feed forward block
        self.ffn_norm=RMSNorm(args.dim,eps=args.norm_eps)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        #(B,seq_len,dim)+(B,swq_len,dim)->(B,seq_len,dim)
        h=x+self.attention.forward(self.attention_norm(x),start_pos,freqs_complex)
        out=h+self.feed_forward.forward(self.ffn_norm(h))
        return out

    
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