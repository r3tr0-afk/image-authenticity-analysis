import torch
import torch.nn as nn

from src.model_clip import CLIPBranch
from src.model_frequency import FrequencyBranch
from src.model_noise import NoiseResidualBranch
from src.model_stats_n_color import StatsColorBranch
from src.model_fusion import GatedFusion


class FullForensicModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        self.clip_branch = CLIPBranch().to(device)
        self.freq_branch = FrequencyBranch().to(device)
        self.noise_branch = NoiseResidualBranch().to(device)
        self.stats_branch = StatsColorBranch().to(device)

        self.fusion = GatedFusion().to(device)

    def forward(self, x):
        f_clip = self.clip_branch(x)
        f_freq = self.freq_branch(x)
        f_noise = self.noise_branch(x)
        f_stats = self.stats_branch(x)

        logit, alphas = self.fusion(
            f_clip=f_clip,
            f_freq=f_freq,
            f_noise=f_noise,
            f_stats=f_stats,
        )

        return logit, alphas
