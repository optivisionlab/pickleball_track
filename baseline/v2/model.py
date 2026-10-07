from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoEncoder(nn.Module):
    """
    Dataset input: (B,C,T,H,W)
    HF VideoMAE input: (B,T,C,H,W)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video):
        if video.ndim != 5:
            raise ValueError(f"Video must be 5-D, got {video.shape}")

        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(pixel_values=video).last_hidden_state


class VideoTemporalAdapter(nn.Module):
    def __init__(self, video_model):
        super().__init__()
        cfg = video_model.config

        self.num_frames = int(cfg.num_frames)
        self.tubelet_size = int(cfg.tubelet_size)
        self.image_size = int(cfg.image_size)
        self.patch_size = int(cfg.patch_size)

        self.num_temporal_tokens = (
            self.num_frames // self.tubelet_size
        )
        self.num_spatial_tokens = (
            self.image_size // self.patch_size
        ) ** 2

    def forward(self, x):
        B, N, D = x.shape

        expected = (
            self.num_temporal_tokens
            * self.num_spatial_tokens
        )

        if N != expected:
            raise ValueError(
                f"VideoMAE token mismatch: got {N}, expected {expected}"
            )

        x = x.reshape(
            B,
            self.num_temporal_tokens,
            self.num_spatial_tokens,
            D,
        )

        # Spatial mean pooling: [B,T,S,D] -> [B,T,D].
        return x.mean(dim=2)


class TemporalMaskAdapter(nn.Module):
    """
    Input/output:
        True  = padding
        False = valid
    """

    def __init__(self, video_model):
        super().__init__()

        cfg = video_model.config
        self.num_frames = int(cfg.num_frames)
        self.tubelet_size = int(cfg.tubelet_size)
        self.num_temporal_tokens = (
            self.num_frames // self.tubelet_size
        )

    def forward(self, frame_mask):
        if frame_mask is None:
            return None

        frame_mask = frame_mask.bool()

        if frame_mask.shape[-1] == self.num_temporal_tokens:
            return frame_mask

        if frame_mask.shape[-1] != self.num_frames:
            raise ValueError(
                f"frame_mask length={frame_mask.shape[-1]}, "
                f"expected {self.num_frames} or "
                f"{self.num_temporal_tokens}"
            )

        B = frame_mask.shape[0]

        mask = frame_mask.reshape(
            B,
            self.num_temporal_tokens,
            self.tubelet_size,
        )

        return mask.all(dim=-1)


class AudioEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, padding_mask):
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        if (
            padding_mask is not None
            and padding_mask.ndim == 3
            and padding_mask.shape[1] == 1
        ):
            padding_mask = padding_mask.squeeze(1)

        # BEATs preprocessing/fbank is kept in fp32.
        audio = audio.float()

        if padding_mask is not None:
            if padding_mask.ndim == 1:
                padding_mask = padding_mask.unsqueeze(0)

            if padding_mask.shape[0] != audio.shape[0]:
                raise ValueError(
                    "audio and padding_mask batch mismatch: "
                    f"audio={tuple(audio.shape)} "
                    f"mask={tuple(padding_mask.shape)}"
                )

            no_valid = padding_mask.all(dim=-1)

            if no_valid.all():
                emb_dim = getattr(
                    getattr(self.model, "cfg", None),
                    "encoder_embed_dim",
                    768,
                )

                feat = torch.zeros(
                    audio.shape[0],
                    1,
                    emb_dim,
                    device=audio.device,
                    dtype=torch.float32,
                )

                out_mask = torch.ones(
                    audio.shape[0],
                    1,
                    dtype=torch.bool,
                    device=audio.device,
                )

                return feat, out_mask

            if no_valid.any():
                valid_rows = ~no_valid
                audio_valid = audio[valid_rows]
                pad_valid = padding_mask[valid_rows]

                with torch.autocast(
                    device_type=audio.device.type,
                    enabled=False,
                ):
                    feat_valid, feat_mask = (
                        self.model.extract_features(
                            audio_valid,
                            padding_mask=pad_valid,
                        )
                    )

                base_feat = torch.zeros(
                    (
                        audio.shape[0],
                        feat_valid.shape[1],
                        feat_valid.shape[-1],
                    ),
                    device=audio.device,
                    dtype=feat_valid.dtype,
                )

                base_mask = torch.ones(
                    (
                        audio.shape[0],
                        feat_valid.shape[1],
                    ),
                    device=audio.device,
                    dtype=torch.bool,
                )

                base_feat[valid_rows] = feat_valid
                base_mask[valid_rows] = feat_mask

                return base_feat, base_mask

        with torch.autocast(
            device_type=audio.device.type,
            enabled=False,
        ):
            return self.model.extract_features(
                audio,
                padding_mask=padding_mask,
            )


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=768, dropout=0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_len, dim)
        )

        nn.init.trunc_normal_(
            self.pos_embedding,
            std=0.02,
        )

    def forward(self, x):
        if x.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(
                "Temporal sequence exceeds positional encoding length"
            )

        return x + self.pos_embedding[:, :x.shape[1]]


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, heads=16, dropout=0.1):
        super().__init__()

        self.cross = nn.MultiheadAttention(
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
        audio,
        audio_padding_mask=None,
    ):
        if audio_padding_mask is not None:
            no_audio = audio_padding_mask.all(dim=1)
        else:
            no_audio = torch.zeros(
                video.shape[0],
                dtype=torch.bool,
                device=video.device,
            )

        fused = video.clone()
        valid_rows = ~no_audio

        if valid_rows.any():
            out, _ = self.cross(
                query=video[valid_rows],
                key=audio[valid_rows],
                value=audio[valid_rows],
                key_padding_mask=(
                    audio_padding_mask[valid_rows]
                    if audio_padding_mask is not None
                    else None
                ),
            )

            fused[valid_rows] = out.to(fused.dtype)

        fused = video + self.dropout(fused)
        return self.norm(fused)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim=768,
        heads=12,
        layers=2,
        dropout=0.1,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, temporal_padding_mask=None):
        if temporal_padding_mask is not None:
            all_padding = temporal_padding_mask.all(dim=1)
        else:
            all_padding = torch.zeros(
                x.shape[0],
                dtype=torch.bool,
                device=x.device,
            )

        out = torch.zeros_like(x)
        valid_rows = ~all_padding

        if valid_rows.any():
            out[valid_rows] = self.transformer(
                x[valid_rows],
                src_key_padding_mask=(
                    temporal_padding_mask[valid_rows]
                    if temporal_padding_mask is not None
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

        self.head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class AVModel(nn.Module):
    """
    The VideoMAE and BEATs pretrained backbones are ALWAYS frozen
    and ALWAYS kept in eval() mode.

    Only:
        video_proj
        audio_proj
        fusion
        temporal_pos
        temporal
        head
    are trainable.
    """

    def __init__(
        self,
        video_model,
        audio_model,
        fusion_dim=768,
        cross_heads=16,
        temporal_heads=12,
        temporal_layers=2,
        temporal_dropout=0.1,
        head_hidden_dim=256,
        head_dropout=0.3,
        num_classes=3,
    ):
        super().__init__()

        self.video_encoder = VideoEncoder(video_model)
        self.video_temporal_adapter = VideoTemporalAdapter(
            video_model
        )
        self.temporal_mask_adapter = TemporalMaskAdapter(
            video_model
        )

        video_dim = int(video_model.config.hidden_size)

        self.audio_encoder = AudioEncoder(audio_model)

        if hasattr(audio_model, "cfg") and hasattr(
            audio_model.cfg,
            "encoder_embed_dim",
        ):
            audio_dim = int(
                audio_model.cfg.encoder_embed_dim
            )
        elif hasattr(audio_model, "config") and hasattr(
            audio_model.config,
            "hidden_size",
        ):
            audio_dim = int(
                audio_model.config.hidden_size
            )
        else:
            audio_dim = 768

        self.video_proj = Projection(
            video_dim,
            fusion_dim,
        )
        self.audio_proj = Projection(
            audio_dim,
            fusion_dim,
        )

        self.fusion = CrossAttentionFusion(
            fusion_dim,
            cross_heads,
            temporal_dropout,
        )

        tokens = (
            self.video_temporal_adapter
            .num_temporal_tokens
        )

        self.temporal_pos = TemporalPositionalEncoding(
            tokens,
            fusion_dim,
        )

        self.temporal = TemporalTransformer(
            fusion_dim,
            temporal_heads,
            temporal_layers,
            temporal_dropout,
        )

        self.head = TemporalPredictionHead(
            fusion_dim,
            head_hidden_dim,
            num_classes,
            head_dropout,
        )

        self.freeze_pretrained_backbones()

    def freeze_pretrained_backbones(self):
        """Permanently freeze pretrained VideoMAE and BEATs."""
        for param in self.video_encoder.parameters():
            param.requires_grad_(False)

        for param in self.audio_encoder.parameters():
            param.requires_grad_(False)

        # Critical: frozen pretrained models never enter train mode.
        self.video_encoder.eval()
        self.audio_encoder.eval()

    def train(self, mode: bool = True):
        """
        Normal train/eval behavior for trainable layers, while
        pretrained backbones are forced to eval() at all times.
        """
        super().train(mode)

        self.video_encoder.eval()
        self.audio_encoder.eval()

        # Guard against accidental unfreezing.
        for param in self.video_encoder.parameters():
            param.requires_grad_(False)

        for param in self.audio_encoder.parameters():
            param.requires_grad_(False)

        return self

    def forward(self, batch, return_mask=False):
        video = batch["video"]
        audio = batch["audio"]
        frame_mask = batch.get("frame_mask")
        audio_mask = batch.get("audio_mask")

        # Backbones are frozen, so do not build an autograd graph through them.
        with torch.no_grad():
            video_feat = self.video_encoder(video)
            video_feat = self.video_temporal_adapter(
                video_feat
            )

            audio_feat, audio_padding_mask = (
                self.audio_encoder(
                    audio,
                    audio_mask,
                )
            )

        video_feat = self.video_proj(video_feat)
        audio_feat = self.audio_proj(audio_feat)

        fused = self.fusion(
            video_feat,
            audio_feat,
            audio_padding_mask,
        )

        fused = self.temporal_pos(fused)

        temporal_padding_mask = (
            self.temporal_mask_adapter(frame_mask)
        )

        temporal_feat = self.temporal(
            fused,
            temporal_padding_mask,
        )

        logits = self.head(temporal_feat)

        if return_mask:
            return logits, temporal_padding_mask

        return logits


class TemporalFocalLoss(nn.Module):
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

    def forward(
        self,
        logits,
        labels,
        temporal_padding_mask=None,
    ):
        B, T, C = logits.shape

        logits = logits.reshape(B * T, C)
        labels = labels.reshape(B * T).long()

        if temporal_padding_mask is not None:
            valid = ~temporal_padding_mask.reshape(
                B * T
            ).bool()

            logits = logits[valid]
            labels = labels[valid]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(
            1,
            labels.unsqueeze(1),
        ).squeeze(1)

        pt = log_pt.exp()

        focal = (
            -(1.0 - pt).pow(self.gamma)
            * log_pt
        )

        focal = focal * self.alpha.to(
            logits.device
        )[labels]

        return focal.mean()