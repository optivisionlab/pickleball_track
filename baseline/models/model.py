import torch
import torch.nn as nn


def get_hidden_dim(model):
    cfg = getattr(model, "config", None) or getattr(model, "cfg", None)
    cfg_attrs = ["hidden_size", "d_model", "dim", "encoder_embed_dim", "encoder_dim", "embed_dim", "embed"]
    if cfg is not None:
        for attr in cfg_attrs:
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
    for attr in cfg_attrs:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise RuntimeError("Cannot infer hidden dimension.")

class VideoEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video):
        return self.model(pixel_values=video).last_hidden_state

class AudioEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, padding_mask):
        feat = self.model.extract_features(audio, padding_mask=padding_mask)
        if isinstance(feat, tuple):
            feat, padding_mask = feat
        elif isinstance(feat, list):
            feat = feat[-1]
        if isinstance(feat, list):
            feat = feat[-1]
        return feat, padding_mask

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=768): 
        super().__init__()
        self.proj=nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    def forward(self,x): 
        return self.proj(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, heads=16):
        super().__init__()
        self.cross = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, video, audio, audio_mask):
        fused, _ = self.cross(
            query=video,
            key=audio,
            value=audio,
            key_padding_mask=audio_mask
        )
        return self.norm(video + fused)

class ClassificationHead(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.mlp=nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    def forward(self, x): 
        x=x.transpose(1, 2)
        x=self.pool(x)
        return self.mlp(x.squeeze(-1))

class AVModel(nn.Module):
    def __init__(self,video_model,audio_model,fusion_dim=768):
        super().__init__()
        self.video_encoder = VideoEncoder(video_model)
        self.audio_encoder=AudioEncoder(audio_model)
        video_dim=get_hidden_dim(video_model)
        audio_dim=get_hidden_dim(audio_model)
        self.video_proj=Projection(video_dim, fusion_dim)
        self.audio_proj=Projection(audio_dim, fusion_dim)
        self.fusion=CrossAttentionFusion(dim=fusion_dim, heads=16)
        self.head=ClassificationHead(fusion_dim)
        
    def forward(self, batch):
        video = batch["video"]
        audio = batch["audio"]
        audio_mask = batch["audio_mask"]
        video_feat = self.video_encoder(video)
        audio_feat, audio_padding_mask = self.audio_encoder(audio, audio_mask)
        video_feat = self.video_proj(video_feat)
        audio_feat = self.audio_proj(audio_feat)
        fused = self.fusion(video_feat, audio_feat, audio_padding_mask)
        return self.head(fused)
