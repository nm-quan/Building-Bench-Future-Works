"""Model registry: 3 official BuildingsBench persistence baselines + LSTM/GRU
+ XGBoost/LightGBM + 7 canonical published forecasting architectures
(PatchTST, iTransformer, TimeXer, DLinear, Informer, Autoformer, Crossformer).

Every neural model shares the forward signature `forward(self, yh, exo) ->
(mu_n, raw_scale, mean, std)`:
  yh   : (B, L) raw load history (kW)
  exo  : (B, L+H, c_exo) = [calendar(N_TIME) | weather(N_WEATHER)? | building_type(1)]
  mu_n, raw_scale : (B, H) normalized-space mean and pre-softplus scale
  mean, std       : (B, 1) RevIN stats to unnormalize mu_n back to kW

Hyperparameters for the published architectures (PatchTST/iTransformer/
TimeXer/Informer/Autoformer/Crossformer) are scaled down from their official
repos' defaults to fit a compute-bounded single-Colab-session budget (12
models x 2 conditions x 200 epochs) -- each docstring notes the official
default alongside the scaled-down value actually used here.
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

# LightGBM (via sklearn's MultiOutputRegressor) warns on every predict() call
# that the input lacks the feature names it was fit with -- harmless (we
# always pass the same fixed-width feature matrix), but noisy at the scale
# this study calls predict().
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------
class RevIN(nn.Module):
    """Reversible Instance Normalization. x:(B,L) -> (normalized, mean, std)."""
    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True).clamp_min(1e-3)
        return (x - mu) / sd, mu, sd


def split_exo(exo, n_time, n_weather, use_weather):
    """exo layout: [calendar(n_time) | weather(n_weather)? | building_type(1)]"""
    cal = exo[..., :n_time]
    weather = exo[..., n_time:n_time + n_weather] if use_weather else None
    btype = exo[..., -1:]
    return cal, weather, btype


def c_exo_width(n_time, n_weather, use_weather):
    return n_time + (n_weather if use_weather else 0) + 1


# ===========================================================================
# 1) Official BuildingsBench persistence baselines (closed-form, 0 params).
#    Ported from buildings_bench/models/persistence.py -- AveragePersistence
#    has a natural sigma (std across the 7 same-hour context values);
#    CopyLastDay/CopyLastWeek do NOT (the official predict() returns sigma=
#    None for these), so we return sigma=None too and train.py skips CRPS
#    for them, matching the paper's own treatment exactly.
# ===========================================================================
class AveragePersistence(nn.Module):
    """mu[t+i] = mean of load at t+i-24j for j=1..7 (same hour, each of the
    past 7 days in the 168h context); sigma = std of those same 7 values."""
    def __init__(self, L=168, H=24, **kw):
        super().__init__()
        self.L, self.H = L, H

    def forward(self, yh, exo):
        grid = yh.view(yh.size(0), self.L // 24, 24)  # (B, 7, 24): day j, hour-of-context h
        mu = grid.mean(1)                               # (B, 24)
        sd = grid.std(1).clamp_min(1e-3)
        gmu = yh.mean(1, keepdim=True)
        gsd = yh.std(1, keepdim=True).clamp_min(1e-3)
        mu_n = (mu - gmu) / gsd
        raw = torch.log(torch.exp(sd / gsd) - 1 + 1e-6)  # invert softplus so softplus(raw)*gsd ~= sd
        return mu_n, raw, gmu, gsd


class CopyLastDayPersistence(nn.Module):
    """mu[t+i] = load at t+i-24 (yesterday, same hour). No sigma (matches
    the official model's predict() -> (forecast, None))."""
    def __init__(self, L=168, H=24, **kw):
        super().__init__()
        assert L >= 24 and H <= 24
        self.L, self.H = L, H

    def forward(self, yh, exo):
        mu = yh[:, self.L - 24: self.L - 24 + self.H]
        gmu = yh.mean(1, keepdim=True)
        gsd = yh.std(1, keepdim=True).clamp_min(1e-3)
        mu_n = (mu - gmu) / gsd
        return mu_n, None, gmu, gsd


class CopyLastWeekPersistence(nn.Module):
    """mu[t+i] = load at t+i-168 (same hour, 7 days ago). No sigma."""
    def __init__(self, L=168, H=24, **kw):
        super().__init__()
        assert L >= 168 and H <= 24
        self.L, self.H = L, H

    def forward(self, yh, exo):
        mu = yh[:, self.L - 168: self.L - 168 + self.H]
        gmu = yh.mean(1, keepdim=True)
        gsd = yh.std(1, keepdim=True).clamp_min(1e-3)
        mu_n = (mu - gmu) / gsd
        return mu_n, None, gmu, gsd


# ===========================================================================
# 2) LSTM / GRU -- DeepAR-style encoder-decoder (RNN encodes L past values,
#    final hidden state + future-covariate projection -> direct H-step head).
# ===========================================================================
class _RNNBase(nn.Module):
    def __init__(self, cell, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 hidden=96, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.revin = RevIN()
        RNN = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = RNN(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, hidden), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, 2 * H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        out, _ = self.rnn(yn.unsqueeze(-1))
        h = out[:, -1]
        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        o = self.head(torch.cat([h, fut], -1)).view(B, self.H, 2)
        return o[..., 0], o[..., 1], mu, sd


class LSTM(_RNNBase):
    def __init__(self, **kw): super().__init__("lstm", **kw)


class GRU(_RNNBase):
    def __init__(self, **kw): super().__init__("gru", **kw)


# ===========================================================================
# 3) DLinear -- decomposition (moving-avg trend + seasonal residual) + linear.
#    Official repo: cnhzcy123/DLinear (kernel_size=25 default) -- matches.
# ===========================================================================
class DLinear(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 kernel=25, dh=128, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H, self.k = L, H, kernel
        self.revin = RevIN()
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.trend = nn.Linear(L, H)
        self.seasonal = nn.Linear(L, H)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, dh), nn.GELU(), nn.Linear(dh, H))
        self.sig = nn.Sequential(nn.Linear(L + H * c_exo, dh), nn.GELU(), nn.Linear(dh, H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        tr = F.avg_pool1d(F.pad(yn.unsqueeze(1), (self.k // 2, self.k // 2), mode="replicate"), self.k, 1).squeeze(1)
        se = yn - tr
        fut = exo[:, self.L:].reshape(B, -1)
        mu_n = self.trend(tr) + self.seasonal(se) + self.fut(fut)
        return mu_n, self.sig(torch.cat([yn, fut], -1)), mu, sd


# ===========================================================================
# 4) PatchTST -- channel-independent patching + Transformer encoder.
#    Official repo (yuqinie98/PatchTST) electricity config: patch_len=16,
#    stride=8, d_model=128, e_layers=3, n_heads=16, seq_len=336 -- kept
#    patch_len/stride/e_layers/n_heads as-is; d_model unchanged (already
#    modest); seq_len follows our L=168. No native exogenous path in the
#    official model (purely channel-independent on the target series), so
#    future covariates are injected via a separate projection head, same
#    pattern used for every other model in this registry.
# ===========================================================================
class PatchTST(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 patch_len=16, stride=8, d_model=128, e_layers=3, n_heads=16,
                 d_ff=256, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (L - patch_len) // stride + 1
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, e_layers)
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(self.n_patches * d_model + d_model),
                                   nn.Linear(self.n_patches * d_model + d_model, 2 * H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        patches = self.patch_embed(yn.unfold(1, self.patch_len, self.stride)) + self.pos
        z = self.encoder(patches).reshape(B, -1)
        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        out = self.head(torch.cat([z, fut], -1)).view(B, self.H, 2)
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# 5) iTransformer -- inverted embedding: each variate (load, each calendar
#    channel, each weather channel, building-type) becomes ONE token via a
#    linear embedding of its full history; attention mixes across variates.
#    Official repo (thuml/iTransformer) ECL config: d_model=512, e_layers=3
#    -- d_model scaled down to 128 for compute budget; e_layers kept at 3.
# ===========================================================================
class ITransformer(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=128, e_layers=3, n_heads=8, d_ff=256, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.n_time, self.n_weather, self.use_weather = n_time, n_weather, use_weather
        self.revin = RevIN()
        self.load_tok = nn.Linear(L, d_model)
        self.cal_tok = nn.ModuleList([nn.Linear(L, d_model) for _ in range(n_time)])
        self.weather_tok = nn.ModuleList([nn.Linear(L, d_model) for _ in range(n_weather)]) if use_weather else None
        self.btype_tok = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, e_layers)
        self.load_head = nn.Linear(d_model, 2 * H)

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        cal_hist = exo[:, :self.L, :self.n_time]
        toks = [self.load_tok(yn)] + [self.cal_tok[i](cal_hist[..., i]) for i in range(self.n_time)]
        if self.use_weather:
            w_hist = exo[:, :self.L, self.n_time:self.n_time + self.n_weather]
            toks += [self.weather_tok[i](w_hist[..., i]) for i in range(self.n_weather)]
        toks.append(self.btype_tok(exo[:, 0, -1:]))
        var_tokens = torch.stack(toks, 1)               # (B, n_var, d_model)
        z = self.encoder(var_tokens)
        out = self.load_head(z[:, 0]).view(B, self.H, 2)  # load token (index 0) carries the forecast
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# 6) TimeXer -- patch tokens (endogenous, self-attention) + variate tokens
#    (exogenous) bridged into ONE learnable global token via cross-attention.
#    Official repo (thuml/TimeXer) ECL config: d_model=512, e_layers 1-3 --
#    d_model scaled to 128, e_layers=2, matching the paper's endo/exo split
#    exactly (this is the one architecture built for our exact use case).
# ===========================================================================
class TimeXer(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=128, n_heads=8, layers=2, d_ff=256, patch_len=24, stride=12, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.n_time, self.n_weather, self.use_weather = n_time, n_weather, use_weather
        self.revin = RevIN()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (L - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.global_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.endo_enc = nn.TransformerEncoder(enc_layer, layers)
        self.cal_tok = nn.ModuleList([nn.Linear(L, d_model) for _ in range(n_time)])
        self.weather_tok = nn.ModuleList([nn.Linear(L, d_model) for _ in range(n_weather)]) if use_weather else None
        self.btype_tok = nn.Linear(1, d_model)
        self.exo_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.exo_norm = nn.LayerNorm(d_model)
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 2 * H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        patches = self.patch_embed(yn.unfold(1, self.patch_len, self.stride)) + self.pos
        seq = torch.cat([self.global_tok.expand(B, -1, -1), patches], 1)
        seq = self.endo_enc(seq)
        g = seq[:, 0:1]                                  # bridging global token
        cal_hist = exo[:, :self.L, :self.n_time]
        toks = [self.cal_tok[i](cal_hist[..., i]) for i in range(self.n_time)]
        if self.use_weather:
            w_hist = exo[:, :self.L, self.n_time:self.n_time + self.n_weather]
            toks += [self.weather_tok[i](w_hist[..., i]) for i in range(self.n_weather)]
        toks.append(self.btype_tok(exo[:, 0, -1:]))
        exo_tok = torch.stack(toks, 1)
        bridged, _ = self.exo_cross(g, exo_tok, exo_tok)
        fused = self.exo_norm(g + bridged).squeeze(1)
        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        out = self.head(torch.cat([fused, fut], -1)).view(B, self.H, 2)
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# 7) Informer -- ProbSparse self-attention (only the top-u "active" queries,
#    ranked by a sparsity measurement, attend densely; the rest get the mean
#    value) + self-attention distillation (halving sequence length between
#    encoder layers via strided conv). Official repo (zhouhaoyi/Informer2020)
#    core mechanism ported faithfully; direct H-step head (not the
#    autoregressive generative decoder) to match this registry's convention.
# ===========================================================================
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads, self.factor = n_heads, factor
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, H, dh).transpose(1, 2)  # (B,H,L,dh)
        k = k.view(B, L, H, dh).transpose(1, 2)
        v = v.view(B, L, H, dh).transpose(1, 2)

        u = min(L, max(1, int(self.factor * math.ceil(math.log(max(L, 2))))))
        # sparsity measurement M(q_i) = max_j(q_i.k_j) - mean_j(q_i.k_j); sample a subset
        # of keys to estimate it cheaply (Informer's own approximation).
        sample_k = min(L, max(1, int(self.factor * math.ceil(math.log(max(L, 2))))))
        idx = torch.randint(0, L, (sample_k,), device=x.device)
        k_sample = k[:, :, idx, :]                                  # (B,H,sample_k,dh)
        qk_sample = torch.einsum("bhld,bhsd->bhls", q, k_sample)    # (B,H,L,sample_k)
        sparsity = qk_sample.max(-1).values - qk_sample.mean(-1)    # (B,H,L)
        top_idx = sparsity.topk(u, dim=-1).indices                  # (B,H,u)

        q_top = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, dh))  # (B,H,u,dh)
        scores = torch.einsum("bhud,bhld->bhul", q_top, k) / math.sqrt(dh)         # (B,H,u,L)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out_top = torch.einsum("bhul,bhld->bhud", attn, v)          # (B,H,u,dh)

        # non-selected queries fall back to the mean value (Informer's default)
        out = v.mean(2, keepdim=True).expand(-1, -1, L, -1).clone()
        out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, dh), out_top)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class InformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DistillConv(nn.Module):
    """Halves the sequence length between encoder layers (Informer's
    self-attention distillation)."""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        z = F.elu(self.conv(x.transpose(1, 2)))
        return self.pool(z).transpose(1, 2)


class Informer(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=128, n_heads=8, e_layers=3, d_ff=256, factor=5, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.revin = RevIN()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        self.layers = nn.ModuleList([InformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.distill = nn.ModuleList([DistillConv(d_model) for _ in range(e_layers - 1)])
        final_len = L
        for _ in range(e_layers - 1):
            final_len = math.ceil(final_len / 2)
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(final_len * d_model + d_model),
                                   nn.Linear(final_len * d_model + d_model, 2 * H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        z = self.embed(yn.unsqueeze(-1)) + self.pos
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.distill):
                z = self.distill[i](z)
        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        out = self.head(torch.cat([z.reshape(B, -1), fut], -1)).view(B, self.H, 2)
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# 8) Autoformer -- series decomposition (moving-avg trend/seasonal, applied
#    BETWEEN layers, not just once) + Auto-Correlation (FFT-based period
#    discovery replacing dot-product attention). Official repo
#    (thuml/Autoformer) core mechanism ported faithfully.
# ===========================================================================
class SeriesDecomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.k = kernel

    def forward(self, x):  # x: (B, L, D)
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (self.k // 2, self.k // 2), mode="replicate"),
                              self.k, 1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """FFT-based period discovery: correlate q and k via FFT, pick the
    top-k lag periods by correlation strength, aggregate v at those lags
    with softmax-normalized weights (time-delay aggregation)."""
    def __init__(self, d_model, n_heads, factor=1, dropout=0.1):
        super().__init__()
        self.n_heads, self.factor = n_heads, factor
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, H, dh).permute(0, 2, 3, 1)  # (B,H,dh,L)
        k = k.view(B, L, H, dh).permute(0, 2, 3, 1)
        v = v.view(B, L, H, dh).permute(0, 2, 3, 1)

        q_f = torch.fft.rfft(q.float(), dim=-1)
        k_f = torch.fft.rfft(k.float(), dim=-1)
        corr = torch.fft.irfft(q_f * torch.conj(k_f), n=L, dim=-1)  # (B,H,dh,L) autocorrelation per lag
        corr_mean = corr.mean(dim=2)                                 # (B,H,L) averaged over channels

        top_k = max(1, int(self.factor * math.log(max(L, 2))))
        weights, delays = corr_mean.topk(top_k, dim=-1)               # (B,H,top_k)
        weights = F.softmax(weights, dim=-1)

        # Vectorized time-delay aggregation (no per-sample/per-head Python loop):
        # for each of the top_k lags, build a (B,H,L) gather index representing
        # a per-(batch,head) circular shift of v along the time axis, then
        # combine with softmax weights.
        t = torch.arange(L, device=x.device).view(1, 1, 1, L)          # (1,1,1,L)
        shift = delays.unsqueeze(-1)                                    # (B,H,top_k,1)
        idx = (t - shift) % L                                           # (B,H,top_k,L)
        idx = idx.unsqueeze(2).expand(-1, -1, dh, -1, -1)               # (B,H,dh,top_k,L)
        v_exp = v.unsqueeze(3).expand(-1, -1, -1, top_k, -1)            # (B,H,dh,top_k,L)
        v_shifted = torch.gather(v_exp, -1, idx)                        # (B,H,dh,top_k,L)
        out = (v_shifted * weights.view(B, H, 1, top_k, 1)).sum(3)      # (B,H,dh,L)
        out = out.permute(0, 3, 1, 2).reshape(B, L, D)
        return self.dropout(self.out(out))


class AutoformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, kernel=25):
        super().__init__()
        self.autocorr = AutoCorrelation(d_model, n_heads, dropout=dropout)
        self.decomp1 = SeriesDecomp(kernel)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.decomp2 = SeriesDecomp(kernel)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        season, _ = self.decomp1(x + self.autocorr(x))
        season2, _ = self.decomp2(season + self.ff(season))
        return season2


class Autoformer(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=96, n_heads=8, e_layers=2, d_ff=192, kernel=25, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.revin = RevIN()
        self.decomp_init = SeriesDecomp(kernel)
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        self.layers = nn.ModuleList([AutoformerLayer(d_model, n_heads, d_ff, dropout, kernel) for _ in range(e_layers)])
        self.trend_proj = nn.Linear(L, H)  # trend component carried through with a simple linear extrapolation
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(L * d_model + d_model), nn.Linear(L * d_model + d_model, H))
        self.scale_head = nn.Sequential(nn.LayerNorm(L * d_model), nn.Linear(L * d_model, H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        seasonal, trend = self.decomp_init(yn.unsqueeze(-1))
        z = self.embed(seasonal) + self.pos
        for layer in self.layers:
            z = layer(z)
        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        seasonal_out = self.head(torch.cat([z.reshape(B, -1), fut], -1))
        trend_out = self.trend_proj(trend.squeeze(-1))
        mu_n = seasonal_out + trend_out
        raw_scale = self.scale_head(z.reshape(B, -1))
        return mu_n, raw_scale, mu, sd


# ===========================================================================
# 9) Crossformer -- Dimension-Segment-Wise (DSW) embedding + stacked
#    Two-Stage Attention (cross-time self-attn per variate, then
#    router-based cross-dimension gather/distribute, O(D) not O(D^2)).
#    Official repo (Thinklab-SJTU/Crossformer) core mechanism.
# ===========================================================================
class TwoStageAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, n_seg, D, n_routers=4, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 2
        self.time_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.time_norm1 = nn.LayerNorm(d_model)
        self.time_ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.time_norm2 = nn.LayerNorm(d_model)
        self.routers = nn.Parameter(torch.randn(1, n_seg, n_routers, d_model) * 0.02)
        self.gather_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.distrib_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.dim_norm1 = nn.LayerNorm(d_model)
        self.dim_ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dim_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B, D, n_seg, d)
        B, D, n_seg, d = x.shape
        xt = x.reshape(B * D, n_seg, d)
        a, _ = self.time_attn(xt, xt, xt)
        xt = self.time_norm1(xt + a)
        xt = self.time_norm2(xt + self.time_ff(xt))
        x = xt.reshape(B, D, n_seg, d)

        xd = x.permute(0, 2, 1, 3).reshape(B * n_seg, D, d)
        r = self.routers.expand(B, -1, -1, -1).reshape(B * n_seg, -1, d)
        gathered, _ = self.gather_attn(r, xd, xd)
        distributed, _ = self.distrib_attn(xd, gathered, gathered)
        xd = self.dim_norm1(xd + distributed)
        xd = self.dim_norm2(xd + self.dim_ff(xd))
        return xd.reshape(B, n_seg, D, d).permute(0, 2, 1, 3)


class Crossformer(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=96, n_heads=8, layers=2, seg_len=24, n_routers=4, dropout=0.15, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.n_time, self.n_weather, self.use_weather = n_time, n_weather, use_weather
        self.revin = RevIN()
        self.seg = seg_len
        self.n_seg = L // seg_len
        self.D = 1 + n_time + (n_weather if use_weather else 0) + 1
        self.seg_embed = nn.Linear(seg_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.D, self.n_seg, d_model) * 0.02)
        self.tsa = nn.ModuleList([TwoStageAttentionLayer(d_model, n_heads, self.n_seg, self.D, n_routers, dropout=dropout)
                                   for _ in range(layers)])
        self.pool = nn.Linear(self.D * self.n_seg * d_model, d_model)
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 2 * H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        cal = exo[:, :self.L, :self.n_time].transpose(1, 2)
        parts = [yn.unsqueeze(1), cal]
        if self.use_weather:
            w = exo[:, :self.L, self.n_time:self.n_time + self.n_weather].transpose(1, 2)
            parts.append(w)
        bt = exo[:, :self.L, -1:].transpose(1, 2)
        parts.append(bt)
        allvar = torch.cat(parts, 1)
        segs = allvar.reshape(B, self.D, self.n_seg, self.seg)
        z = self.seg_embed(segs) + self.pos
        for layer in self.tsa:
            z = layer(z)
        pooled = self.pool(z.reshape(B, -1))
        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        out = self.head(torch.cat([pooled, fut], -1)).view(B, self.H, 2)
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# Registry
# ===========================================================================
REG = {
    "persistence_avg": AveragePersistence,
    "persistence_last_day": CopyLastDayPersistence,
    "persistence_last_week": CopyLastWeekPersistence,
    "lstm": LSTM,
    "gru": GRU,
    "dlinear": DLinear,
    "patchtst": PatchTST,
    "itransformer": ITransformer,
    "timexer": TimeXer,
    "informer": Informer,
    "autoformer": Autoformer,
    "crossformer": Crossformer,
}

MODEL_KW = {
    "persistence_avg": dict(),
    "persistence_last_day": dict(),
    "persistence_last_week": dict(),
    "lstm": dict(hidden=96, layers=2, dropout=0.1),
    "gru": dict(hidden=96, layers=2, dropout=0.1),
    "dlinear": dict(kernel=25, dh=128, dropout=0.1),
    "patchtst": dict(patch_len=16, stride=8, d_model=128, e_layers=3, n_heads=16, d_ff=256, dropout=0.1),
    "itransformer": dict(d_model=128, e_layers=3, n_heads=8, d_ff=256, dropout=0.1),
    "timexer": dict(d_model=128, n_heads=8, layers=2, d_ff=256, patch_len=24, stride=12, dropout=0.1),
    "informer": dict(d_model=128, n_heads=8, e_layers=3, d_ff=256, factor=5, dropout=0.1),
    "autoformer": dict(d_model=96, n_heads=8, e_layers=2, d_ff=192, kernel=25, dropout=0.1),
    "crossformer": dict(d_model=96, n_heads=8, layers=2, seg_len=24, n_routers=4, dropout=0.15),
}


def build(name: str, **kw):
    return REG[name](**kw)


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


# ===========================================================================
# XGBoost / LightGBM -- not nn.Module; share the same (yh, exo) -> flattened
# feature convention (normalized history + future covariates), multi-output
# tree regression on RevIN-normalized targets. This is a practical adaptation
# of BuildingsBench's own LightGBM baseline (skforecast ForecasterAutoreg with
# 168h autoregressive lags + optional temperature exogenous), not a literal
# port of `scripts/transfer_learning_lightgbm.py` -- documented deviation for
# a unified eval harness across all 12 models.
# ===========================================================================
def _tree_xy(yh: torch.Tensor, exo: torch.Tensor, L: int):
    mu = yh.mean(1, keepdim=True)
    sd = yh.std(1, keepdim=True).clamp_min(1e-3)
    yn = (yh - mu) / sd
    X = torch.cat([yn, exo[:, L:].reshape(yh.size(0), -1)], -1).cpu().numpy()
    return X, mu.squeeze(1).cpu().numpy(), sd.squeeze(1).cpu().numpy()


def tree_fit(model_type: str, gather_fn, ds: dict, dev: str, use_w: bool, L: int, H: int,
             n_windows: int = 80000, n_estimators: int = 500, max_depth: int = 6, hours: int = None, win: int = None):
    """model_type: 'xgboost' or 'lightgbm'."""
    hours = hours or ds.get("T", config.HOURS)
    win = win or config.WIN
    smax = hours - win
    b = torch.randint(0, ds["N"], (n_windows,), device=dev)
    s = torch.randint(0, smax, (n_windows,), device=dev)
    yh, yf, exo = gather_fn(ds, b, s, use_w)
    X, mu, sd = _tree_xy(yh, exo, L)
    Y = (yf.cpu().numpy() - mu.reshape(-1, 1)) / sd.reshape(-1, 1)
    k = int(len(X) * 0.9)

    if model_type == "xgboost":
        import xgboost as xgb
        params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      device="cuda" if dev == "cuda" else "cpu")
        try:
            m = xgb.XGBRegressor(multi_strategy="multi_output_tree", **params)
            m.fit(X[:k], Y[:k])
        except Exception:
            from sklearn.multioutput import MultiOutputRegressor
            m = MultiOutputRegressor(xgb.XGBRegressor(**params))
            m.fit(X[:k], Y[:k])
    elif model_type == "lightgbm":
        import lightgbm as lgb
        from sklearn.multioutput import MultiOutputRegressor
        base = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, verbosity=-1)
        m = MultiOutputRegressor(base)
        m.fit(X[:k], Y[:k])
    else:
        raise ValueError(model_type)

    sigma = np.maximum((Y[k:] - m.predict(X[k:])).std(0), 1e-3)
    return m, sigma


def tree_predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
