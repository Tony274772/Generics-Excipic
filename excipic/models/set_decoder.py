"""
Autoregressive Set Decoder with Pointer Attention.

Generates an ordered set of excipients autoregressively.
At each step, attends to:
    1. The fusion context (API + dosage)
    2. The full excipient embedding matrix (knowledge graph output)
And outputs a pointer distribution over excipients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerTransformerDecoder(nn.Module):
    """
    Transformer decoder with pointer-network attention for set generation.

    Architecture:
        - Excipient embedding input (from vocabulary)
        - N transformer decoder layers:
            - Masked self-attention over generated sequence
            - Cross-attention to fusion context
        - Pointer attention head: produces distribution over excipient vocabulary

    Generation:
        - Starts with [BOS]
        - Autoregressively predicts excipients
        - Stops at [EOS] or max_seq_len
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = config.max_seq_len

        # Positional encoding for decoder input sequence
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Projection from excipient embedding dim to decoder hidden dim
        self.input_proj = nn.Linear(512, config.hidden_dim)  # excipient emb dim = 512

        # Context projection: fusion_dim (1536) → hidden_dim
        self.context_proj = nn.Linear(1536, config.hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_layers,
        )

        # Pointer attention head
        self.pointer_query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.pointer_key_proj = nn.Linear(512, config.hidden_dim)  # from excipient emb

        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        fusion_context: torch.Tensor,
        excipient_embeddings: torch.Tensor,
        excipient_lookup_fn,
    ) -> torch.Tensor:
        """
        Forward pass for teacher-forced training.

        Args:
            decoder_input_ids: [B, T] excipient indices ([BOS] + target sequence)
            fusion_context: [B, 1536] fusion vector
            excipient_embeddings: [V, 512] all excipient embeddings
            excipient_lookup_fn: function(indices) → embeddings [*, 512]

        Returns:
            logits: [B, T, V] pointer distribution over excipient vocabulary
        """
        B, T = decoder_input_ids.shape
        device = decoder_input_ids.device

        # 1. Embed input tokens using the excipient embedding lookup
        token_emb = excipient_lookup_fn(decoder_input_ids)  # [B, T, 512]
        token_emb = self.input_proj(token_emb)  # [B, T, hidden_dim]

        # 2. Add positional encoding
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)  # [B, T, hidden_dim]
        token_emb = token_emb + pos_emb

        # 3. Project fusion context as memory for cross-attention
        context = self.context_proj(fusion_context)  # [B, hidden_dim]
        memory = context.unsqueeze(1)  # [B, 1, hidden_dim]

        # 4. Causal mask for self-attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device
        )

        # 5. Transformer decoder
        hidden = self.transformer_decoder(
            tgt=token_emb,
            memory=memory,
            tgt_mask=causal_mask,
        )  # [B, T, hidden_dim]

        hidden = self.output_norm(hidden)

        # 6. Pointer attention: dot-product between decoder hidden and excipient keys
        queries = self.pointer_query_proj(hidden)  # [B, T, hidden_dim]
        keys = self.pointer_key_proj(excipient_embeddings)  # [V, hidden_dim]

        # Scaled dot-product attention
        scale = self.hidden_dim ** 0.5
        logits = torch.matmul(queries, keys.transpose(0, 1)) / scale  # [B, T, V]

        return logits

    @torch.no_grad()
    def generate(
        self,
        fusion_context: torch.Tensor,
        excipient_embeddings: torch.Tensor,
        excipient_lookup_fn,
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        temperature: float = 0.7,
        max_len: int = 15,
    ) -> torch.Tensor:
        """
        Greedy/temperature-based autoregressive generation.

        Args:
            fusion_context: [B, 1536]
            excipient_embeddings: [V, 512]
            excipient_lookup_fn: function(indices) → embeddings
            bos_idx, eos_idx, pad_idx: special token indices
            temperature: Sampling temperature
            max_len: Maximum generation length

        Returns:
            generated_ids: [B, max_len] generated excipient indices
        """
        B = fusion_context.size(0)
        device = fusion_context.device

        # Start with [BOS]
        generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            # Forward pass
            logits = self.forward(
                decoder_input_ids=generated,
                fusion_context=fusion_context,
                excipient_embeddings=excipient_embeddings,
                excipient_lookup_fn=excipient_lookup_fn,
            )

            # Get logits for last position
            next_logits = logits[:, -1, :]  # [B, V]

            # Mask PAD and BOS
            next_logits[:, pad_idx] = float("-inf")
            next_logits[:, bos_idx] = float("-inf")

            # Mask already generated (no duplicates)
            for b in range(B):
                already_generated = generated[b].tolist()
                for idx in already_generated:
                    if idx != bos_idx:
                        next_logits[b, idx] = float("-inf")

            # Temperature sampling
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [B, 1]
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            # Replace with PAD for finished sequences
            next_token[finished] = pad_idx

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_idx)

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        # Remove BOS token
        return generated[:, 1:]

    @torch.no_grad()
    def beam_search(
        self,
        fusion_context: torch.Tensor,
        excipient_embeddings: torch.Tensor,
        excipient_lookup_fn,
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        beam_size: int = 5,
        max_len: int = 15,
    ) -> torch.Tensor:
        """
        Beam search decoding (single sample, B=1).

        Returns:
            Best sequence [max_len] of excipient indices.
        """
        device = fusion_context.device
        assert fusion_context.size(0) == 1, "Beam search expects B=1"

        # Initialize beam
        sequences = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
        scores = torch.zeros(1, device=device)
        finished_seqs = []
        finished_scores = []

        for step in range(max_len):
            n_beams = sequences.size(0)

            # Expand fusion context for all beams
            ctx = fusion_context.expand(n_beams, -1)

            logits = self.forward(
                decoder_input_ids=sequences,
                fusion_context=ctx,
                excipient_embeddings=excipient_embeddings,
                excipient_lookup_fn=excipient_lookup_fn,
            )

            next_logits = logits[:, -1, :]  # [n_beams, V]
            next_logits[:, pad_idx] = float("-inf")
            next_logits[:, bos_idx] = float("-inf")

            # Mask already generated
            for b in range(n_beams):
                for idx in sequences[b].tolist():
                    if idx != bos_idx:
                        next_logits[b, idx] = float("-inf")

            log_probs = F.log_softmax(next_logits, dim=-1)  # [n_beams, V]

            # Total scores
            total_scores = scores.unsqueeze(-1) + log_probs  # [n_beams, V]
            total_scores = total_scores.view(-1)  # [n_beams * V]

            # Top-k
            topk_scores, topk_indices = total_scores.topk(beam_size)
            beam_indices = topk_indices // self.vocab_size
            token_indices = topk_indices % self.vocab_size

            # Build new sequences
            new_seqs = torch.cat([
                sequences[beam_indices],
                token_indices.unsqueeze(-1)
            ], dim=1)

            # Check for EOS
            active_mask = token_indices != eos_idx
            for i in range(beam_size):
                if not active_mask[i]:
                    finished_seqs.append(new_seqs[i])
                    finished_scores.append(topk_scores[i].item())

            if active_mask.sum() == 0:
                break

            sequences = new_seqs[active_mask]
            scores = topk_scores[active_mask]

        # Return best finished sequence (or best active if none finished)
        if finished_seqs:
            best_idx = max(range(len(finished_scores)), key=lambda i: finished_scores[i])
            result = finished_seqs[best_idx][1:]  # remove BOS
        else:
            result = sequences[0, 1:]  # remove BOS

        # Pad to max_len
        if result.size(0) < max_len:
            pad = torch.full(
                (max_len - result.size(0),), pad_idx,
                dtype=torch.long, device=device
            )
            result = torch.cat([result, pad])
        else:
            result = result[:max_len]

        return result.unsqueeze(0)  # [1, max_len]
