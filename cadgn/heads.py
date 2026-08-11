from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
In the methodology, there is one EncoderHead and DecoderHead per tokenizer. 
EncoderHead: LLM Embeddings -> (num_nodes, ca_dgn_dim)
                Accepts pre-embedded token vectors from the LLM's own embedding layer. Position embedding and attention 
                pooling enable effective compression before the sequence is sent to CA-DGN 
DecoderHead: (num_nodes, ca_dgn_dim) -> (num_nodes, seq_len, vocab_size)
                Not Auto-regressive. By using positional embeddings, we can unroll the compressed vector to retrieve the full token sequence.
                All positions predicted simultaneously from one forward pass.
"""


class EncoderHead(nn.Module):
    """
    Encodes pre-embedded node text into the CA-DGN's feature space
    Pipeline:
        embeds (num_nodes, seq_len, llm_dim) -> h (num_nodes, seq_len, ca_dgn_dim)
        h -> pooled (num_nodes, ca_dgn_dim) -> H (num_nodes, ca_dgn_dim)
    """

    def __init__(self,
                 llm_dim: int,
                 ca_dgn_dim: int,
                 max_seq_len: int,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.ca_dgn_dim = ca_dgn_dim
        self.max_seq_len = max_seq_len

        # Where the dimensions of the llm do not match the CA-DGN, we project them.
        self.input_proj = (
            nn.Linear(llm_dim, ca_dgn_dim)
            if llm_dim != ca_dgn_dim
            else nn.Identity()
        )

        # Positional Embedding
        self.pos_embed = nn.Embedding(max_seq_len, ca_dgn_dim)

        self.dropout = nn.Dropout(p=dropout)

        # Attention Pooling - no bias as the positional embedding already provides a positional offset; produces scalar attention logits
        self.pool_attn = nn.Linear(ca_dgn_dim, 1, bias=False)

        # Output Projection
        # two-layer MLP ensures the pooled representation is projected into a space that the CA-DGN encoder can process
        self.proj = nn.Sequential(
            nn.Linear(ca_dgn_dim, ca_dgn_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ca_dgn_dim*2, ca_dgn_dim),
            nn.LayerNorm(ca_dgn_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.pool_attn.weight)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,
                embeds: torch.Tensor,
                attention_mask: torch.Tensor,
                ) -> torch.Tensor:
        """
        :param embeds: (num_nodes, seq_len, llm_dim) FloatTensor Output of the LLM's embedding layer (i.e. `llm.get_input_embeddings()(token_ids)`)
        :param attention_mask: (num_nodes, seq_len) LongTensor or BoolTensor. 1/True for real tokens, 0 / False for padding. Matches attention_mask produced by HuggingFace tokenizers directly.
        :return: H (num_nodes, ca_dgn_dim) Float Tensor of embeddings
        """
        print(embeds.shape)
        N,S, llm_dim = embeds.shape
        device = embeds.device

        # Project to CA-DGN Dim
        h = self.input_proj(embeds)
        # (num_nodes, seq_len, ca_dgn_dim)

        # Add Positional Embeddings
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        # (1, seq_len)   [list of token sequences]

        h = self.dropout(h + self.pos_embed(pos_ids))
        # (num_nodes, seq_len, ca_dgn_dim)

        # Construct Padding Mask
        # (1= real token, 0 = padding), invert so True = positions to ignore
        pad_mask = attention_mask == 0
        # (num_nodes, seq_len)


        # Attention-weighted pooling
        attn_logits = self.pool_attn(h).squeeze(-1)
        # (num_nodes, seq_len)

        attn_logits  = attn_logits.masked_fill(pad_mask, float('-inf'))
        attn_weights = attn_logits.softmax(dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)
        # (num_nodes, ca_dgn_dim)

        # Project to CA-DGN feature space
        return self.proj(pooled)
        # (num_nodes, ca_dgn_dim)




class DecoderHead(nn.Module):
    """
    Reconstructs LLM embedding vectors from CA-DGN feature vectors.

    It is not auto-regressive, as we can construct all token positions from the single fixed vector (given position embedding).
    Functionally, this is equivalent to the EncoderHead in reverse, making this setup an Auto-Encoding framework (cf. Wasserstein Auto-Encoder)
    """

    def __init__(
            self,
            ca_dgn_dim: int,
            llm_dim: int,
            max_seq_len: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.ca_dgn_dim = ca_dgn_dim
        self.llm_dim = llm_dim
        self.max_seq_len = max_seq_len

        # Position queries (one learned vector per sequence position)
        self.pos_queries = nn.Embedding(max_seq_len, ca_dgn_dim)

        # Context Projection (project z into same space as pos_queries, producing a position-independent content vector which is broadcast across all positions)
        self.context_proj = nn.Sequential(
            nn.Linear(ca_dgn_dim, ca_dgn_dim),
            nn.LayerNorm(ca_dgn_dim),
        )

        # Position-wise MLP (Applied identically to all positions as shared weight)
        self.mlp = nn.Sequential(
            nn.Linear(ca_dgn_dim, ca_dgn_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ca_dgn_dim * 4, ca_dgn_dim),
            nn.Dropout(p=dropout)
        )

        self.mlp_norm = nn.LayerNorm(ca_dgn_dim)
        self.out_norm = nn.LayerNorm(ca_dgn_dim)

        # Output projection

        self.out_proj = (
            nn.Linear(ca_dgn_dim, llm_dim)
            if llm_dim != ca_dgn_dim
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_queries.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                z: torch.Tensor,
                attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """

        :param z: (num_nodes, ca_dgn_dim) output of CADGNDecoder
        :param attention_mask: (num_nodes, seq_len) LongTensor or BoolTensor, 1 for real tokens, 0 for padding. When provided, S is taken from the mask, rather than max_seq_len, reconstructing only real positions. When None (inference) falls back to max_seq_len.
        :return: embeds (num_nodes, max_seq_len, llm_dim) Reconstructed LLM embedding vectors. Compare against EncoderHead's input embeddings via MSE or cosine similarity.
        """
        N = z.size(0)
        S = attention_mask.size(1) if attention_mask is not None else self.max_seq_len
        device = z.device

        # Position Queries
        pos_ids = torch.arange(S, device=device)
        pos_emb = self.pos_queries(pos_ids).unsqueeze(0).expand(N, -1, -1)
        # (num_nodes, seq_len, ca_dgn_dim)

        # Context from z
        ctx = self.context_proj(z).unsqueeze(1).expand(-1, S, -1)
        # (num_nodes, seq_len, ca_dgn_dim)

        # Additive position conditioning
        h = pos_emb + ctx
        # (num_nodes, seq_len, ca_dgn_dim)

        # MLP with pre-norm + residual
        h = h + self.mlp(self.mlp_norm(h))
        # (num_nodes, seq_len, ca_dgn_dim)

        # Output projection -> LLM embedding space
        return self.out_proj(self.out_norm(h))
        # (num_nodes, seq_len, llm_dim)
