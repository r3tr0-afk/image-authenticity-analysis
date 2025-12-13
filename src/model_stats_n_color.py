import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_ycbcr(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=1)


def rgb_to_hsv(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    rc = (maxc - r) / (maxc - minc + 1e-6)
    gc = (maxc - g) / (maxc - minc + 1e-6)
    bc = (maxc - b) / (maxc - minc + 1e-6)
    h = torch.zeros_like(maxc)
    h[maxc == r] = (bc - gc)[maxc == r]
    h[maxc == g] = 2.0 + (rc - bc)[maxc == g]
    h[maxc == b] = 4.0 + (gc - rc)[maxc == b]
    return (h / 6.0) % 1.0, s, v


def moments(x):
    mean = x.mean(dim=[1,2])
    std = x.std(dim=[1,2])
    xc = x - mean[:, None, None]
    skew = (xc**3).mean(dim=[1,2]) / (std**3 + 1e-6)
    kurt = (xc**4).mean(dim=[1,2]) / (std**4 + 1e-6)
    return mean, std, skew, kurt


def ycbcr_cov(ycbcr):
    B = ycbcr.shape[0]
    v = ycbcr.view(B, 3, -1)
    v = v - v.mean(dim=2, keepdim=True)
    cov = torch.bmm(v, v.transpose(1,2)) / (v.shape[2] - 1)
    return torch.stack([
        cov[:,0,0], cov[:,0,1], cov[:,0,2],
        cov[:,1,1], cov[:,1,2],
        cov[:,2,2]
    ], dim=1)


def laplacian_variance(x):
    kernel = torch.tensor(
        [[0,1,0],[1,-4,1],[0,1,0]],
        device=x.device,
        dtype=x.dtype
    ).view(1,1,3,3)
    lap = F.conv2d(x.unsqueeze(1), kernel, padding=1)
    return lap.var(dim=[1,2,3])


def extract_patches(x, grid=4):
    B, H, W = x.shape
    ps = H // grid
    return [
        x[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps]
        for i in range(grid) for j in range(grid)
    ]


def patch_self_similarity(patches):
    vecs = []
    for p in patches:
        v = F.interpolate(p.unsqueeze(1), (8,8), mode="bilinear", align_corners=False)
        vecs.append(F.normalize(v.flatten(1), dim=1))
    V = torch.stack(vecs, dim=1)
    sims = torch.bmm(V, V.transpose(1,2))
    mask = ~torch.eye(sims.size(1), device=sims.device).bool()
    sims = sims[:, mask].view(sims.size(0), -1)
    return torch.stack([
        sims.mean(1),
        sims.std(1),
        sims.max(1).values,
        (sims > 0.9).float().mean(1),
        sims.topk(3, dim=1).values.mean(1)
    ], dim=1)


def hue_entropy(h):
    B = h.size(0)
    out = []
    for i in range(B):
        hist = torch.histc(h[i], bins=12, min=0, max=1)
        hist = hist / (hist.sum() + 1e-6)
        out.append(-(hist * torch.log(hist + 1e-6)).sum())
    return torch.stack(out)


def gradient_channel_corr(x):
    gx = torch.gradient(x, dim=2)[0]
    gy = torch.gradient(x, dim=3)[0]
    g = torch.sqrt(gx**2 + gy**2)
    corr = []
    for i in range(x.size(0)):
        rg = torch.corrcoef(torch.stack([
            g[i,0].flatten(), g[i,1].flatten()
        ]))[0,1]
        rb = torch.corrcoef(torch.stack([
            g[i,0].flatten(), g[i,2].flatten()
        ]))[0,1]
        gb = torch.corrcoef(torch.stack([
            g[i,1].flatten(), g[i,2].flatten()
        ]))[0,1]
        corr.append((rg + rb + gb) / 3)
    return torch.stack(corr)


class StatsColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(39, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

    def forward(self, x):
        feats = []

        for c in range(3):
            feats.extend(moments(x[:,c]))

        ycbcr = rgb_to_ycbcr(x)
        feats.append(ycbcr_cov(ycbcr))
        feats.append(ycbcr[:,1].mean([1,2]))
        feats.append(ycbcr[:,2].mean([1,2]))

        y, cb, cr = ycbcr[:,0], ycbcr[:,1], ycbcr[:,2]
        py = extract_patches(y)
        pcb = extract_patches(cb)
        pcr = extract_patches(cr)

        for arr in [
            torch.stack([p.mean([1,2]) for p in py],1),
            torch.stack([p.std([1,2]) for p in py],1),
            torch.stack([laplacian_variance(p) for p in py],1),
            torch.stack([p.std([1,2]) for p in pcb],1),
            torch.stack([p.std([1,2]) for p in pcr],1)
        ]:
            feats.append(arr.mean(1))
            feats.append(arr.std(1))

        feats.append(patch_self_similarity(py))

        h, s, _ = rgb_to_hsv(x)
        feats.append(hue_entropy(h))
        feats.append(s.mean([1,2]))
        feats.append(s.std([1,2]))
        feats.append(gradient_channel_corr(x))

        feat_vec = torch.cat([f.unsqueeze(1) if f.ndim == 1 else f for f in feats], dim=1)
        return self.projector(feat_vec)
