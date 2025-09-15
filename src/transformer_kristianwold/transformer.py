import torch
import torch.nn.functional as F
import torch.nn as nn
import textwrap
import ipywidgets as widgets
from IPython.display import display
from torch.distributions import Categorical

import numpy as np
import math

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        tf_blocks,
        embed_dim,
        heads,
        ff_dim,
        dropout=0.1,
        start_token_id=None,
        use_weight_tying=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.tf_blocks = tf_blocks
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.start_token_id = start_token_id
        self.use_weight_tying = use_weight_tying

        self.head_dim = embed_dim // heads
        self.dol = nn.Dropout(dropout)

        # embedding
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        k = 1/math.sqrt(vocab_size)
        torch.nn.init.normal_(self.word_embed.weight, mean=0.0, std=k)
        torch.nn.init.normal_(self.pos_embed.weight, mean=0.0, std=k)

        # transformer layers
        self.layer_list = nn.ModuleList([TransformerBlock(vocab_size, max_seq_len, heads, embed_dim, ff_dim, dropout, start_token_id) for _ in range(tf_blocks)])

        # unembedding
        if not use_weight_tying:
            self.word_unembed = nn.Linear(embed_dim, vocab_size, bias=False).weight

        self.unembed_b = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))  # bias for unembedding
    
    def forward(self, tokens):
        batch, seq = tokens.shape[0], tokens.shape[1]
        mask = get_causal_mask(tokens, self.start_token_id).unsqueeze(1)  # [batch, 1, seq, seq]
        mask = mask.expand(batch, self.heads, seq, seq)

        x = self.embed(tokens)
        for block in self.layer_list:
            x = block(x, mask)
        
        x = self.unembed(x)

        return x
    
    def embed(self, tokens):
        seq = tokens.shape[1]
        if seq > self.max_seq_len:
            tokens = tokens[:, -self.max_seq_len :] # truncate to max_seq_len
            seq = self.max_seq_len

        x_embeds = self.word_embed(tokens)  # [batch, seq, embed_dim]

        pos_ids = resetting_positions(tokens, self.start_token_id) # reset position indices at start tokens
        pos_embeds = self.pos_embed(pos_ids)

        x_embeds = x_embeds + pos_embeds
        x_embeds = self.dol(x_embeds) # dropout embeddings

        return x_embeds

    def unembed(self, x_embeds):
        if self.use_weight_tying:
            word_unembed = self.word_embed.weight
        else:
            word_unembed = self.word_unembed

        logits = x_embeds @ word_unembed.T + self.unembed_b

        return logits
    
    def num_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_word_embed(self):
        return self.word_embed.weight




class TransformerBlock(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        heads,
        embed_dim,
        ff_dim,
        dropout,
        start_token_id,

    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.start_token_id = start_token_id

        self.head_dim = embed_dim // heads

        self.dol1 = nn.Dropout(dropout)
        self.dol2 = nn.Dropout(dropout)
        self.do_attn = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.KQV = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.WO = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layer_up = nn.Linear(embed_dim, ff_dim, bias=True)
        self.layer_down = nn.Linear(ff_dim, embed_dim, bias=True)


    def forward(self, x_embeds, mask):
        x_embeds = self.attention(x_embeds, mask)

        x_embeds = self.ffnn(x_embeds)

        return x_embeds


    def attention(self, x_embeds, mask):
        batch = x_embeds.shape[0]
        seq = x_embeds.shape[1]

        # compute keys, queries and values
        x_kqv = self.KQV(x_embeds)  # [batch, seq, 3*heads*head_dim]
        x_kqv = torch.reshape(x_kqv, [batch, seq, 3, self.heads, self.head_dim])
        x_kqv = torch.permute(x_kqv, [0, 3, 2, 1, 4])

        x_k = x_kqv[:, :, 0, :, :] # keys, [batch, heads, seq, head_dim]
        x_q = x_kqv[:, :, 1, :, :] # queries
        x_v = x_kqv[:, :, 2, :, :] # values

        # compute attention
        attn = torch.matmul(x_q, x_k.transpose(-1, -2))  # [batch, heads, seq, seq]
        attn = attn / math.sqrt(self.head_dim)  # scale attention scores


        attn_masked = attn.masked_fill(mask, float("-inf"))  # [batch, heads, seq, seq]
        attn_masked = F.softmax(attn_masked, dim=-1)  # softmax over the last dimension
        attn_masked = self.do_attn(attn_masked)  # dropout 

        # compute weighted output
        out = torch.matmul(attn_masked, x_v)  # [batch, heads, seq, head_dim]
        out = torch.permute(out, [0, 2, 1, 3])
        out = torch.reshape(out, [batch, seq, self.embed_dim])
        out = self.WO(out)  # [batch, seq, embed_dim]
    
        # apply dropout, layer norm and residual connection

        out = self.dol1(out)
        out = self.ln1(out + x_embeds)

        return out
    
    def ffnn(self, x_embeds):
        """
        Feed forward neural network (FFNN)
        """

        out = self.layer_up(x_embeds) # scale up
        out = F.relu(out)             # nonlinearity  
        out = self.layer_down(out)    # scale down   '

        # apply dropout, layer norm and residual connection 
        out = self.dol2(out)
        out = self.ln2(out + x_embeds)

        return out


def get_causal_mask(tokens, start_token_id):
    triu_mask = get_triu_mask(tokens)
    block_mask = get_block_diag_mask(tokens, start_token_id)
    causal_mask = triu_mask | block_mask

    return causal_mask


def get_triu_mask(tokens):
    """
    Returns a triangular mask for the given tokens.
    """
    batch_size, seq_len = tokens.size()
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
    mask = mask.to(tokens.device)  # Move mask to the same device as tokens

    return mask


def get_block_diag_mask(tokens, start_token_id):
    """
    Returns a block diagonal mask for cutting attention across start tokens.
    """
    is_start = torch.eq(tokens, start_token_id)
    segment_ids = torch.cumsum(is_start, axis=1)
    seg_i = torch.unsqueeze(segment_ids, 2)                          
    seg_j = torch.unsqueeze(segment_ids, 1)                        
    mask   = ~torch.eq(seg_i, seg_j).to(tokens.device)#  [batch,  seq, seq]                                                      

    return mask


def resetting_positions(tokens, start_token_id):
    """
    tokens:          int Tensor of shape [batch, seq_len]
    start_token_id:  scalar int — the ID of your “start” token
    returns:
      rel_pos:       int Tensor of shape [batch, seq_len],
                     where rel_pos[b,i] counts up from 0 since
                     the last start‐token (or the sequence start).
    """

    is_start = tokens.eq(start_token_id)                              # [B, T], bool

    # set beginning of each sequence as start
    batch_size, seq_len = tokens.size()
    first_col = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
    is_start = torch.cat([first_col, is_start[:, 1:]], dim=1)         # [B, T]

    # initialize default [0,1,2,...,T-1] positions for each batch
    positions = torch.arange(seq_len, dtype=torch.long, device=tokens.device)  # [T]
    positions = positions.unsqueeze(0).expand(batch_size, -1)         # [B, T]

    # 4) pick out the positions where resets happen, else 0
    start_pos = torch.where(is_start, positions, torch.zeros_like(positions))  # [B, T]

    # torch.cummax finds latest reset position along each sequence
    last_start, _ = start_pos.cummax(dim=1)                          # [B, T]

    # 6) subtract to get positions relative to each last reset
    rel_pos = positions - last_start                                 # [B, T]
    return rel_pos




class Inference:
    def __init__(self, model, tokenizer, context_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device
        

    def run(self, text, T, k, mode=None):
        if mode == "summary":
            text = "<s><b>" + text + "<h>"
        elif mode == "expand":
            text = "<s><h>" + text + "<b>"
        else:
            pass

        tokens = torch.tensor(self.tokenizer.encode(text.lower()), dtype=torch.long).reshape(1, -1).to(self.device)

        self.display = Display()

        self.model.eval()
        with torch.no_grad():
            for i in range(self.context_length):
                next = self.next_token(tokens, T, k,)

                tokens = torch.cat([tokens, next.reshape(1,1)], dim=1)
                text = self.tokenizer.decode(tokens[0].tolist())
                self.display.update(text)

                if next[0] == self.tokenizer.token_to_idx["</s>"]:
                    break
                

    def next_token(self, tokens, T, k):
        logits = self.model(tokens)[0, -1:]
        topk_vals, _    = torch.topk(logits, k=k)
        kth_value       = topk_vals[:,-1]

        logits = torch.where(logits >= kth_value, logits, -torch.inf)
        dist = Categorical(logits=logits/T)
        next = dist.sample()

        return next


class Display:
    def __init__(self):
        self.wrapper = textwrap.TextWrapper(width=80)

        self.ta = widgets.Textarea(
            value="",
            layout=widgets.Layout(width='80ch', height='20em'),
            disabled=True
        )
        display(self.ta)

    def update(self, text):
        self.ta.value = self.wrapper.fill(text.replace("\n", " "))  # this updates in-place
