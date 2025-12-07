import torch
import torch.nn as nn
import clip
from torchvision import transforms as T


class CLIPBranch(nn.Module):
    def __init__(self, device=None, dropout=0.1):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.preprocess = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
        )
        self.projector.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)

        x = self.preprocess(x)

        with torch.no_grad():
            emb = self.clip_model.encode_image(x)

        emb = emb.to(dtype=self.projector[0].weight.dtype)

        out = self.projector(emb)                
        return out
