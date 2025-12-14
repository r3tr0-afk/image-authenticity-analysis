import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj_clip = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )

        self.proj_freq = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )

        self.proj_noise = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )

        self.proj_stats = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )

        self.gate_clip = self._make_gate()
        self.gate_freq = self._make_gate()
        self.gate_noise = self._make_gate()
        self.gate_stats = self._make_gate()

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _make_gate():
        return nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, f_clip, f_freq, f_noise, f_stats):
        p_clip  = self.proj_clip(f_clip)
        p_freq  = self.proj_freq(f_freq)
        p_noise = self.proj_noise(f_noise)
        p_stats = self.proj_stats(f_stats)

        s_clip  = self.gate_clip(p_clip)
        s_freq  = self.gate_freq(p_freq)
        s_noise = self.gate_noise(p_noise)
        s_stats = self.gate_stats(p_stats)

        scores = torch.cat([s_clip, s_freq, s_noise, s_stats], dim=1)
        alphas = F.softmax(scores, dim=1)

        fused = (
            alphas[:, 0:1] * p_clip +
            alphas[:, 1:2] * p_freq +
            alphas[:, 2:3] * p_noise +
            alphas[:, 3:4] * p_stats
        )

        logit = self.classifier(fused)

        return logit, alphas