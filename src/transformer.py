import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def block_diag_mask(tokens, start_token_id):

    is_start = torch.eq(tokens, start_token_id)
    segment_ids = torch.cumsum(is_start, axis=1)
    seg_i = torch.unsqueeze(segment_ids, 2)                          
    seg_j = torch.unsqueeze(segment_ids, 1)                        
    mask   = ~torch.eq(seg_i, seg_j)
    mask = torch.unsqueeze(mask, 1).to(tokens.get_device()) # [batch, 1, seq, seq]                                                        

    return mask


def resetting_positions(tokens: torch.Tensor, start_token_id: int) -> torch.Tensor:
    """
    tokens:          int Tensor of shape [batch, seq_len]
    start_token_id:  scalar int — the ID of your “start” token
    returns:
      rel_pos:       int Tensor of shape [batch, seq_len],
                     where rel_pos[b,i] counts up from 0 since
                     the last start‐token (or the sequence start).
    """
    # 1) detect start tokens
    is_start = tokens.eq(start_token_id)                              # [B, T], bool

    # 2) force a “start” at position 0 of each sequence
    batch_size, seq_len = tokens.size()
    first_col = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
    is_start = torch.cat([first_col, is_start[:, 1:]], dim=1)         # [B, T]

    # 3) make a [0,1,2,…,T-1] index array for each batch
    positions = torch.arange(seq_len, dtype=torch.long, device=tokens.device)  # [T]
    positions = positions.unsqueeze(0).expand(batch_size, -1)         # [B, T]

    # 4) pick out the indices where resets happen (else 0)
    start_pos = torch.where(is_start, positions, torch.zeros_like(positions))  # [B, T]

    # 5) compute, for each token, the **latest** reset‐position seen so far.
    #    torch.cummax returns a tuple (values, indices), we only need values.
    last_start, _ = start_pos.cummax(dim=1)                          # [B, T]

    # 6) subtract to get “position since last reset”
    rel_pos = positions - last_start                                 # [B, T]
    return rel_pos


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

        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.mha = nn.MultiHeadAttention(embed_dim, heads, dropout=dropout, batch_first=True)

        self.layer_up = nn.Linear(embed_dim, ff_dim, bias=True)
        self.layer_down = nn.Linear(ff_dim, embed_dim, bias=True)


    def forward(self, x_embeds, tokens):
        x_embeds = self.attention(x_embeds, tokens)
        x_embeds = self.ffnn(x_embeds)

        return x_embeds


    def attention(self, x_embeds, tokens):
        batch = x_embeds.shape[0]
        seq = x_embeds.shape[1]


        if not self.start_token_id is None:
            block_mask = block_diag_mask(tokens, self.start_token_id)
            attn_mask = future_mask.unsqueeze(0).unsqueeze(0) | block_mask
        else:
            attn_mask = future_mask.unsqueeze(0).unsqueeze(0) 

        attention, _ = self.mha(x_embeds, x_embeds, x_embeds)

        

        attention = nn.functional.scaled_dot_product_attention(x_q, x_k, x_v, 
                                                                dropout_p=self.dropout,
                                                                attn_mask=attn_mask)


        attention = torch.permute(attention, [0, 2, 1, 3])  # [batch, seq, heads, head_dim]
        attention = torch.reshape(attention, [batch, seq, self.embed_dim])
        out = self.WO(attention)  # [batch, seq, embed_dim]
        
        out = self.dol1(out)
        out = self.ln1(out)
        # pre-norm to keep gradients alive
        out = out + x_embeds

        return out
    
    def ffnn(self, x_embeds):
        out = self.layer_up(x_embeds)
        out = F.relu(out)
        out = self.layer_down(out)
        out = self.dol2(out)
        out = self.ln2(out)

        out = out + x_embeds

        return out
    

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
        pad_token_id=None,
        start_token_id=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.tf_blocks = tf_blocks
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id

        self.head_dim = embed_dim // heads
        self.dol = nn.Dropout(dropout)

        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.block_list = nn.ModuleList([TransformerBlock(vocab_size, max_seq_len, heads, embed_dim, ff_dim, dropout, start_token_id) for _ in range(tf_blocks)])

        self.unembed_b = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
    
        
    def forward(self, tokens):

        x, tokens = self.embed(tokens)
            
        for block in self.block_list:
            x = block(x, tokens)
        
        x = self.unembed(x)

        return x
    
    def embed(self, tokens, training=False):
        seq = tokens.shape[1]
        if seq > self.max_seq_len:
            tokens = tokens[:, -self.max_seq_len :]
            seq = self.max_seq_len

        x_embeds = self.word_embed(tokens)  # [batch, seq, embed_dim]

        pos_ids = resetting_positions(tokens, self.start_token_id)
        pos_embeds = self.pos_embed(pos_ids)  # [seq, embed_dim]

        x_embeds = x_embeds + pos_embeds
        x_embeds = self.dol(x_embeds)

        return x_embeds, tokens

    def unembed(self, x_embeds):
        w_embed = torch.transpose(self.word_embed.weight, 0, 1)  # [embed_dim, vocab_size]

        logits = x_embeds @ w_embed + self.unembed_b

        return logits
