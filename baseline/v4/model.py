from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    """
    Project one modality into the common fusion dimension.
    """

    def __init__(self, in_dim, out_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    Video-conditioned cross attention.

    Video is the QUERY.
    Context modality (audio/text) is KEY + VALUE.

    Returns only the attended context, not the video residual.
    The residual is handled explicitly by GatedFusion.
    """

    def __init__(self, dim=768, heads=16, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        video,
        context,
        context_padding_mask=None,
    ):
        """
        Args:
            video:
                [B, Tv, D]

            context:
                [B, Tc, D]

            context_padding_mask:
                [B, Tc], True = padding

        Returns:
            context_features:
                [B, Tv, D]
        """

        B = video.shape[0]

        if context_padding_mask is not None:
            context_padding_mask = context_padding_mask.bool()

            # Avoid MultiheadAttention receiving a sample in which
            # every context token is masked.
            no_context = context_padding_mask.all(dim=1)
        else:
            no_context = torch.zeros(
                B,
                dtype=torch.bool,
                device=video.device,
            )

        out = torch.zeros_like(video)

        valid = ~no_context

        if valid.any():
            attn_out, _ = self.attn(
                query=video[valid],
                key=context[valid],
                value=context[valid],
                key_padding_mask=(
                    context_padding_mask[valid]
                    if context_padding_mask is not None
                    else None
                ),
                need_weights=False,
            )

            out[valid] = self.dropout(
                self.norm(attn_out)
            )

        return out


class GatedFusion(nn.Module):
    """
    Dynamic gated multimodal fusion.

    The original video representation is always preserved:

        F = V + alpha * A_context + beta * T_context

    alpha and beta are learned dynamically for every video timestep.
    """

    def __init__(self, dim=768, dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        video,
        audio_context,
        text_context,
    ):
        """
        Args:
            video:
                [B, Tv, D]

            audio_context:
                [B, Tv, D]

            text_context:
                [B, Tv, D]

        Returns:
            fused:
                [B, Tv, D]
        """

        gate_input = torch.cat(
            [
                video,
                audio_context,
                text_context,
            ],
            dim=-1,
        )

        # [B, Tv, 2]
        gates = torch.sigmoid(
            self.gate(gate_input)
        )

        alpha = gates[..., 0:1]
        beta = gates[..., 1:2]

        fused = (
            video
            + alpha * audio_context
            + beta * text_context
        )

        fused = self.norm(
            fused
        )

        return self.dropout(fused)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, length=16, dim=768):
        super().__init__()

        self.pos = nn.Parameter(
            torch.zeros(1, length, dim)
        )

        nn.init.trunc_normal_(
            self.pos,
            std=0.02,
        )

    def forward(self, x):
        if x.shape[1] > self.pos.shape[1]:
            raise ValueError(
                f"Temporal sequence length {x.shape[1]} "
                f"exceeds positional embedding length "
                f"{self.pos.shape[1]}."
            )

        return x + self.pos[:, :x.shape[1]]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim=768,
        heads=12,
        layers=2,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        if padding_mask is not None:
            padding_mask = padding_mask.bool()

            all_padding = padding_mask.all(
                dim=1
            )
        else:
            all_padding = torch.zeros(
                x.shape[0],
                dtype=torch.bool,
                device=x.device,
            )

        out = torch.zeros_like(x)

        valid = ~all_padding

        if valid.any():
            out[valid] = self.encoder(
                x[valid],
                src_key_padding_mask=(
                    padding_mask[valid]
                    if padding_mask is not None
                    else None
                ),
            )

        return self.norm(out)


class TemporalPredictionHead(nn.Module):
    def __init__(
        self,
        dim=768,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.net(x)


class AVTFeatureModel(nn.Module):
    """
    VIDEO + AUDIO + TEXT TRAINING MODEL.

    This is an extension of the original AVFeatureModel.

    Frozen pretrained encoders are intentionally NOT inside this model:

        - VideoMAE features are precomputed/cached.
        - BEATs features are precomputed/cached.
        - CLIP text features are precomputed/cached.

    Therefore this model contains only trainable modules:

        video_proj
        audio_proj
        text_proj

        video <- audio cross-attention
        video <- text cross-attention

        gated multimodal fusion

        temporal positional embedding
        temporal transformer
        temporal prediction head

    Architecture:

        Video [B,Tv,Dv]
            |
        Video Projection
            |
        V [B,Tv,768]
            |
            +-----------------------------+
            |                             |
            | Query                       | Query
            v                             v
        Cross Attention              Cross Attention
        V <- Audio                    V <- Text
            ^                             ^
            |                             |
        A [B,Ta,768]                  T [B,1,768]
            |                             |
        Audio Projection              Text Projection
            ^                             ^
            |                             |
          BEATs                           CLIP
            ^                             ^
            |                             |
          Audio                           Text

            Audio context + Text context
                       |
                       v
                 Gated Fusion
                       |
                       v
                  [B,Tv,768]
                       |
                       v
            Temporal Positional Encoding
                       |
                       v
              Temporal Transformer
                       |
                       v
                  [B,Tv,768]
                       |
                       v
              Temporal Prediction Head
                       |
                       v
                    [B,Tv,3]

    Classes:
        0 = no_event
        1 = hit
        2 = miss
    """

    def __init__(
        self,
        video_dim,
        audio_dim,
        text_dim,
        fusion_dim=768,
        cross_heads=16,
        temporal_heads=12,
        temporal_layers=2,
        dropout=0.1,
        head_hidden_dim=256,
        head_dropout=0.3,
        num_classes=3,
        temporal_length=16,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Modality projections
        # ---------------------------------------------------------
        self.video_proj = Projection(
            video_dim,
            fusion_dim,
            dropout,
        )

        self.audio_proj = Projection(
            audio_dim,
            fusion_dim,
            dropout,
        )

        self.text_proj = Projection(
            text_dim,
            fusion_dim,
            dropout,
        )

        # ---------------------------------------------------------
        # Video-conditioned cross-modal attention
        # ---------------------------------------------------------
        self.audio_cross_attn = CrossModalAttention(
            dim=fusion_dim,
            heads=cross_heads,
            dropout=dropout,
        )

        self.text_cross_attn = CrossModalAttention(
            dim=fusion_dim,
            heads=cross_heads,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Gated fusion
        #
        # F = V + alpha * A_context + beta * T_context
        # ---------------------------------------------------------
        self.fusion = GatedFusion(
            dim=fusion_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Temporal modeling
        # ---------------------------------------------------------
        self.temporal_pos = TemporalPositionalEncoding(
            length=temporal_length,
            dim=fusion_dim,
        )

        self.temporal = TemporalTransformer(
            dim=fusion_dim,
            heads=temporal_heads,
            layers=temporal_layers,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Temporal prediction
        # ---------------------------------------------------------
        self.head = TemporalPredictionHead(
            dim=fusion_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=head_dropout,
        )

        self._assert_all_trainable()

    def _assert_all_trainable(self):
        trainable = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad
        ]

        if not trainable:
            raise RuntimeError(
                "AVTFeatureModel has no trainable parameters."
            )

    def forward(
        self,
        video_features,
        audio_features,
        audio_padding_mask,
        temporal_padding_mask,
        text_features,
        text_padding_mask=None,
    ):
        """
        Args:
            video_features:
                Precomputed VideoMAE features.

                Shape:
                    [B, Tv, video_dim]

            audio_features:
                Precomputed BEATs features.

                Shape:
                    [B, Ta, audio_dim]

            audio_padding_mask:
                True = padded audio timestep.

                Shape:
                    [B, Ta]

            temporal_padding_mask:
                True = padded video/temporal timestep.

                Shape:
                    [B, Tv]

            text_features:
                Precomputed CLIP text features.

                Accepted shapes:
                    [B, text_dim]
                    [B, 1, text_dim]

            text_padding_mask:
                Optional text padding mask.

                Shape:
                    [B, 1]

                True = padding.

                If omitted, text is considered valid for every sample.

        Returns:
            logits:
                [B, Tv, num_classes]
        """

        # ---------------------------------------------------------
        # Normalize input shapes
        # ---------------------------------------------------------
        if video_features.ndim != 3:
            raise ValueError(
                "video_features must have shape "
                "[B, Tv, video_dim]. "
                f"Got {tuple(video_features.shape)}."
            )

        if audio_features.ndim == 2:
            audio_features = audio_features.unsqueeze(1)

        if audio_features.ndim != 3:
            raise ValueError(
                "audio_features must have shape "
                "[B, Ta, audio_dim] or [B, audio_dim]. "
                f"Got {tuple(audio_features.shape)}."
            )

        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(1)

        if text_features.ndim != 3:
            raise ValueError(
                "text_features must have shape "
                "[B, 1, text_dim] or [B, text_dim]. "
                f"Got {tuple(text_features.shape)}."
            )

        if text_features.shape[1] != 1:
            raise ValueError(
                "For this architecture, text must contain one "
                "global context token per sample: [B, 1, text_dim]. "
                f"Got {tuple(text_features.shape)}."
            )

        # ---------------------------------------------------------
        # Project modalities into common dimension
        # ---------------------------------------------------------
        video = self.video_proj(
            video_features
        )

        audio = self.audio_proj(
            audio_features
        )

        text = self.text_proj(
            text_features
        )

        # ---------------------------------------------------------
        # Video <- Audio
        # ---------------------------------------------------------
        audio_context = self.audio_cross_attn(
            video=video,
            context=audio,
            context_padding_mask=audio_padding_mask,
        )

        # ---------------------------------------------------------
        # Video <- Text
        # ---------------------------------------------------------
        text_context = self.text_cross_attn(
            video=video,
            context=text,
            context_padding_mask=text_padding_mask,
        )

        # ---------------------------------------------------------
        # Gated multimodal fusion
        #
        # V is always preserved as residual.
        # Audio/text contributions are dynamically gated.
        # ---------------------------------------------------------
        x = self.fusion(
            video=video,
            audio_context=audio_context,
            text_context=text_context,
        )

        # ---------------------------------------------------------
        # Temporal modeling
        # ---------------------------------------------------------
        x = self.temporal_pos(x)

        x = self.temporal(
            x,
            temporal_padding_mask,
        )

        # ---------------------------------------------------------
        # Per-temporal-token prediction
        # ---------------------------------------------------------
        logits = self.head(x)

        return logits


class TemporalFocalLoss(nn.Module):
    """
    Same focal-loss formulation as the original baseline.

    alpha:
        [no_event, hit, miss]
    """

    def __init__(
        self,
        alpha=(0.25, 1.0, 1.0),
        gamma=2.0,
    ):
        super().__init__()

        self.gamma = float(gamma)

        self.register_buffer(
            "alpha",
            torch.tensor(
                alpha,
                dtype=torch.float32,
            ),
        )

    def forward(self, logits, labels, padding_mask=None):
        B, T, C = logits.shape

        logits = logits.reshape(
            B * T,
            C,
        )

        labels = labels.reshape(
            B * T
        ).long()

        if padding_mask is not None:
            valid = ~padding_mask.reshape(
                B * T
            ).bool()

            logits = logits[valid]
            labels = labels[valid]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )

        log_pt = log_probs.gather(
            1,
            labels[:, None],
        ).squeeze(1)

        pt = log_pt.exp()

        loss = -(
            (1.0 - pt).pow(self.gamma)
            * log_pt
        )

        loss = loss * self.alpha.to(logits.device)[labels]

        return loss.mean()