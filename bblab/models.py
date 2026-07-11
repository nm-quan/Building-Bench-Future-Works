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
# 10) Transformer-S/M/L (Gaussian) -- the paper's OWN model, not a canonical
#     baseline from elsewhere. Faithful port of buildings_bench/models/
#     transformers.py's LoadForecastingTransformer with continuous_loads=True,
#     continuous_head='gaussian_nll' -- an encoder-decoder nn.Transformer,
#     trained with teacher forcing, autoregressive greedy decoding at
#     inference (matching the official generate_sample(greedy=True)).
#     Hyperparameters below match buildings_bench/configs/TransformerWithGaussian-
#     {S,M,L}.toml exactly. Deviation: the official model also embeds each
#     building's lat/lon (via a PUMA-centroid lookup table this pipeline
#     doesn't carry through) -- zero-embedded here, matching the official
#     model's own `ignore_spatial=True` code path.
#
#     Unlike every other model in this registry, this one does NOT use RevIN:
#     the official model operates directly in Box-Cox+standardized space with
#     no additional per-window instance normalization, so `mean`/`std` here
#     are the fixed identity (0, 1) -- `mu_n`/`raw_scale` ARE the Box-Cox-
#     space prediction, matching the paper's own training/inference convention
#     exactly rather than layering our other models' RevIN choice on top.
# ===========================================================================
class _PositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding, ported verbatim
    from buildings_bench.models.transformers.PositionalEncoding."""
    def __init__(self, d_model, dropout, maxlen=500):
        super().__init__()
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pe = torch.zeros((maxlen, d_model))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (B, T, d_model)
        return self.dropout(x + self.pe[:x.size(1)].unsqueeze(0))


class _ZeroEmbedding(nn.Module):
    """Matches official ZeroEmbedding -- outputs zeros shaped like the
    reference tensor's (batch, seq) dims. Used here for lat/lon, which this
    pipeline's data doesn't carry (see class docstring above)."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.zeros = nn.Parameter(torch.zeros(1, 1, embedding_dim), requires_grad=False)

    def forward(self, ref):  # ref: (B, T, ...) -- only shape[0]/shape[1] used
        return self.zeros.expand(ref.shape[0], ref.shape[1], -1)


class SinusoidalPeriodicEmbedding(nn.Module):
    """Matches official TimeSeriesSinusoidalPeriodicEmbedding: a scalar
    already linearly scaled to [-1,1] -> [sin(pi*x), cos(pi*x)] -> linear."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(2, embedding_dim)

    def forward(self, x):  # x: (B, T)
        with torch.no_grad():
            feats = torch.stack([torch.sin(math.pi * x), torch.cos(math.pi * x)], dim=-1)
        return self.linear(feats)


class TransformerGaussian(nn.Module):
    USES_TEACHER_FORCING = True  # signals train.py's train() to pass yf during training

    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.0, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.n_time, self.n_weather, self.use_weather = n_time, n_weather, use_weather
        s = max(1, d_model // 256)  # official side-embedding width scales with model size

        # Official width is a fixed 64, but 768(L)+64=832 isn't divisible by
        # nhead=12 (PyTorch's MultiheadAttention requires embed_dim % nhead
        # == 0) -- the official d_model is always an exact multiple of nhead
        # on its own (256/4, 512/8, 768/12), so round the weather width to
        # the nearest multiple of nhead too, to keep d_total valid. Matches
        # the official 64 exactly for S/M (16*4, 8*8); L gets 60 instead of
        # 64 (5*12) -- a small, documented deviation forced by a combination
        # (L + weather) that would crash the official code's own numbers too.
        weather_dim = max(nhead, nhead * round(64 / nhead)) if use_weather else 0
        self.weather_embed = nn.Linear(n_weather, weather_dim) if use_weather else None
        d_total = d_model + weather_dim
        self.d_total = d_total

        self.transformer = nn.Transformer(
            d_model=d_total, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True)
        self.power_embed = nn.Linear(1, 64 * s)
        self.logits = nn.Linear(d_total, 2)  # Gaussian: [mean, raw_scale]
        self.pos_enc = _PositionalEncoding(d_total, dropout)
        self.building_embed = nn.Embedding(2, 32 * s)
        self.lat_embed = _ZeroEmbedding(32 * s)
        self.lon_embed = _ZeroEmbedding(32 * s)
        self.doy_embed = SinusoidalPeriodicEmbedding(32 * s)
        self.dow_embed = SinusoidalPeriodicEmbedding(32 * s)
        self.hod_embed = SinusoidalPeriodicEmbedding(32 * s)

    def _embed_series(self, load_bc, cal, weather, btype_idx):
        """load_bc: (B,T) Box-Cox-space load. cal: (B,T,n_time). weather:
        (B,T,n_weather) or None. btype_idx: (B,T) long (0=res, 1=com)."""
        parts = [
            self.lat_embed(load_bc.unsqueeze(-1)), self.lon_embed(load_bc.unsqueeze(-1)),
            self.building_embed(btype_idx),
            self.doy_embed(cal[..., 0]), self.dow_embed(cal[..., 1]), self.hod_embed(cal[..., 2]),
        ]
        if self.use_weather:
            parts.append(self.weather_embed(weather))
        parts.append(self.power_embed(load_bc.unsqueeze(-1)))
        return torch.cat(parts, -1)

    def _split_exo(self, exo):
        cal = exo[..., :self.n_time]
        weather = exo[..., self.n_time:self.n_time + self.n_weather] if self.use_weather else None
        btype_idx = ((exo[..., -1] > 0).long())  # +1(com)->1, -1(res)->0
        return cal, weather, btype_idx

    def forward(self, yh, exo, yf=None):
        B, dev = yh.size(0), yh.device
        cal, weather, btype_idx = self._split_exo(exo)
        zero = torch.zeros(1, 1, device=dev, dtype=yh.dtype)

        if yf is not None:
            # ---- training: teacher forcing, one parallel decoder pass ----
            full_load = torch.cat([yh, yf], 1)                       # (B, L+H) Box-Cox space
            embed = self._embed_series(full_load, cal, weather, btype_idx)
            src = self.pos_enc(embed[:, :self.L])
            tgt = self.pos_enc(embed[:, self.L - 1:-1])               # context's last step + tgt[:-1] (shifted)
            tgt_mask = self.transformer.generate_square_subsequent_mask(self.H).to(dev)
            memory = self.transformer.encoder(src)
            out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
            mu_raw = self.logits(out)                                  # (B, H, 2)
            mu_n, raw_scale = mu_raw[..., 0], mu_raw[..., 1]
        else:
            # ---- inference: autoregressive greedy decoding, matching the
            # official generate_sample(greedy=True) exactly ----
            ctx_embed = self._embed_series(yh, cal[:, :self.L], weather[:, :self.L] if self.use_weather else None,
                                            btype_idx[:, :self.L])
            memory = self.transformer.encoder(self.pos_enc(ctx_embed))
            # Raw (not-yet-positionally-encoded) decoder embeddings, grown one
            # step at a time. Positional encoding is re-applied to the WHOLE
            # growing sequence every iteration (matching the official
            # generate_sample loop exactly -- it re-embeds `decoder_input` in
            # full each step, not just the newest token, since this model's
            # PositionalEncoding always numbers positions 0..len-1 of whatever
            # it's given).
            raw_decoder_embeds = [ctx_embed[:, -1:]]  # seed: context's last raw embedded step
            preds, raws = [], []
            for k in range(1, self.H + 1):
                decoder_input = self.pos_enc(torch.cat(raw_decoder_embeds, 1))
                tgt_mask = self.transformer.generate_square_subsequent_mask(k).to(dev)
                dec_out = self.transformer.decoder(decoder_input, memory, tgt_mask=tgt_mask)
                mu_raw = self.logits(dec_out[:, -1:])                  # (B, 1, 2)
                preds.append(mu_raw[..., 0])
                raws.append(mu_raw[..., 1])
                if k < self.H:
                    step_cal = cal[:, self.L + k - 1: self.L + k]
                    step_w = weather[:, self.L + k - 1: self.L + k] if self.use_weather else None
                    step_bt = btype_idx[:, self.L + k - 1: self.L + k]
                    raw_decoder_embeds.append(self._embed_series(preds[-1], step_cal, step_w, step_bt))
            mu_n = torch.cat(preds, 1)
            raw_scale = torch.cat(raws, 1)

        mean = zero.expand(B, 1)   # identity: no RevIN, see class docstring
        std = torch.ones_like(mean)
        return mu_n, raw_scale, mean, std


# ===========================================================================
# 11) TFT-Lite -- Temporal Fusion Transformer's core mechanism (Lim et al.
#     2019/2021, "Temporal Fusion Transformers for Interpretable Multi-
#     horizon Time Series Forecasting"): every exogenous variable is embedded
#     INDEPENDENTLY, then a learned Variable Selection Network produces a
#     PER-TIMESTEP soft weighting over which variables matter -- absent from
#     every other model in this registry, which either concatenate all
#     covariates uniformly or use fixed (unweighted) per-variable tokens.
#     Followed by an LSTM local-processing layer and self-attention over the
#     full sequence, matching the official encoder-LSTM + interpretable-
#     attention design. Separate encoder/decoder variable sets since the
#     future window has no load value to select over.
# ===========================================================================
class GatedResidualNetwork(nn.Module):
    """GRN(a, c) = LayerNorm(skip(a) + GLU(W1 ELU(W2 a + W3 c))). Matches the
    TFT paper's GRN exactly; context `c` is optional."""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1, has_context=False):
        super().__init__()
        self.w2 = nn.Linear(d_in, d_hidden)
        self.wc = nn.Linear(d_hidden, d_hidden, bias=False) if has_context else None
        self.w1 = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.glu = nn.Linear(d_hidden, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a, c=None):
        eta2 = self.w2(a)
        if self.wc is not None and c is not None:
            eta2 = eta2 + self.wc(c)
        eta2 = F.elu(eta2)
        eta1 = self.dropout(self.w1(eta2))
        gate, val = self.glu(eta1).chunk(2, -1)
        gated = torch.sigmoid(gate) * val
        return self.norm(self.skip(a) + gated)


class VariableSelectionNetwork(nn.Module):
    """n_vars independent scalar inputs, each embedded to d_model, -> a
    per-timestep softmax over the n_vars -> weighted sum -> (B,T,d_model)."""
    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.var_embed = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_vars)])
        self.var_grn = nn.ModuleList([GatedResidualNetwork(d_model, d_model, d_model, dropout) for _ in range(n_vars)])
        self.weight_grn = GatedResidualNetwork(n_vars * d_model, d_model, n_vars, dropout)

    def forward(self, vars_list):  # list of n_vars tensors, each (B,T)
        embeds = [emb(v.unsqueeze(-1)) for emb, v in zip(self.var_embed, vars_list)]   # each (B,T,d_model)
        flat = torch.cat(embeds, -1)
        weights = F.softmax(self.weight_grn(flat), -1)                                  # (B,T,n_vars)
        processed = torch.stack([grn(e) for grn, e in zip(self.var_grn, embeds)], -1)   # (B,T,d_model,n_vars)
        return (processed * weights.unsqueeze(2)).sum(-1)                               # (B,T,d_model)


class TFTLite(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=64, n_heads=4, dropout=0.1, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.n_time, self.n_weather, self.use_weather = n_time, n_weather, use_weather
        self.revin = RevIN()
        n_enc_vars = 1 + n_time + (n_weather if use_weather else 0) + 1   # load + calendar + weather? + btype
        n_dec_vars = n_time + (n_weather if use_weather else 0) + 1        # no load in the future
        self.enc_vsn = VariableSelectionNetwork(n_enc_vars, d_model, dropout)
        self.dec_vsn = VariableSelectionNetwork(n_dec_vars, d_model, dropout)
        self.encoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.gate_enrich = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.pos_ff = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.head = nn.Linear(d_model, 2)

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        cal, weather, btype = split_exo(exo, self.n_time, self.n_weather, self.use_weather)

        enc_vars = [yn] + [cal[:, :self.L, i] for i in range(self.n_time)]
        if self.use_weather:
            enc_vars += [weather[:, :self.L, i] for i in range(self.n_weather)]
        enc_vars.append(btype[:, :self.L, 0])
        enc_repr = self.enc_vsn(enc_vars)                                  # (B,L,d_model)

        dec_vars = [cal[:, self.L:, i] for i in range(self.n_time)]
        if self.use_weather:
            dec_vars += [weather[:, self.L:, i] for i in range(self.n_weather)]
        dec_vars.append(btype[:, self.L:, 0])
        dec_repr = self.dec_vsn(dec_vars)                                  # (B,H,d_model)

        enc_out, (h, c) = self.encoder_lstm(enc_repr)
        dec_out, _ = self.decoder_lstm(dec_repr, (h, c))
        seq = self.gate_enrich(torch.cat([enc_out, dec_out], 1))           # (B,L+H,d_model)

        causal_mask = torch.triu(torch.full((self.L + self.H, self.L + self.H), float("-inf"), device=yh.device), 1)
        attn_out, _ = self.attn(seq, seq, seq, attn_mask=causal_mask)
        seq = self.attn_norm(seq + attn_out)
        seq = self.pos_ff(seq)

        out = self.head(seq[:, -self.H:])                                  # (B,H,2)
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# 12) xLSTM-Patch -- sLSTM cell (Beck et al. 2024, "xLSTM: Extended Long
#     Short-Term Memory": exponential input/forget gates + a log-space
#     stabilizer to prevent overflow) applied over DAILY PATCHES (7 patches
#     of 24h each) rather than raw hourly steps, so the sequential
#     recurrence stays cheap on Colab. Directly motivated by a 2026 benchmark
#     (arXiv:2605.09722, "Benchmarking Transformer and xLSTM for Time-Series
#     Forecasting of Heat Consumption") showing xLSTM beats Transformer/TFT
#     variants specifically on building heat-consumption forecasting.
# ===========================================================================
class SLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Rz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ri = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Rf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Ro = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_state(self, B, device, dtype):
        z = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        return (z, z, z, z)  # h, c, n, m

    def forward(self, x, state):
        h, c, n, m = state
        z = torch.tanh(self.Wz(x) + self.Rz(h))
        i_tilde = self.Wi(x) + self.Ri(h)          # log-input-gate pre-activation
        f_tilde = self.Wf(x) + self.Rf(h)          # log-forget-gate pre-activation
        o = torch.sigmoid(self.Wo(x) + self.Ro(h))
        m_new = torch.maximum(f_tilde + m, i_tilde)  # stabilizer (log-space running max)
        i = torch.exp(i_tilde - m_new)
        f = torch.exp(f_tilde + m - m_new)
        c_new = f * c + i * z
        n_new = f * n + i
        h_new = o * (c_new / n_new.clamp_min(1e-6))
        return h_new, (h_new, c_new, n_new, m_new)


class XLSTMPatch(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=96, patch_len=24, layers=2, dropout=0.1, dh=128, **kw):
        super().__init__()
        self.L, self.H, self.patch_len = L, H, patch_len
        self.n_patches = L // patch_len
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.cells = nn.ModuleList([SLSTMCell(d_model, d_model) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, dh), nn.GELU(), nn.Linear(dh, d_model))
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 2 * H))

    def forward(self, yh, exo):
        B, dev, dtype = yh.size(0), yh.device, yh.dtype
        yn, mu, sd = self.revin(yh)
        x = self.patch_embed(yn.view(B, self.n_patches, self.patch_len))   # (B,7,d_model)

        for cell in self.cells:
            state = cell.init_state(B, dev, dtype)
            hs = []
            for t in range(self.n_patches):
                h, state = cell(x[:, t], state)
                hs.append(h)
            x = self.dropout(torch.stack(hs, 1))
        summary = x[:, -1]                                                  # (B,d_model)

        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        out = self.head(torch.cat([summary, fut], -1)).view(B, self.H, 2)
        return out[..., 0], out[..., 1], mu, sd


# ===========================================================================
# 13) SpectraMix -- TSMixer-style time+feature MLP-mixing (Chen et al. 2023,
#     "TSMixer: An All-MLP Architecture for Time Series Forecasting") over
#     daily patches, fused with a FITS-style (Xu et al. 2024, "FITS: Modeling
#     Time Series with 10k Parameters") frequency-domain linear extrapolation
#     branch, plus an explicit degree-day weather-response correction. No
#     attention, no recurrence -- cheap, and diversifies the registry with a
#     pure-MLP + frequency-domain mechanism.
# ===========================================================================
class _MixerBlock(nn.Module):
    def __init__(self, n_patches, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.time_norm = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(nn.Linear(n_patches, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, n_patches))
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):  # x: (B, n_patches, d_model)
        t = self.time_norm(x).transpose(1, 2)             # (B,d_model,n_patches)
        x = x + self.time_mlp(t).transpose(1, 2)
        x = x + self.feat_mlp(self.feat_norm(x))
        return x


class FrequencyExtrapolation(nn.Module):
    """FITS's core trick: a learned complex-linear layer mapping the rfft of
    a length-L series to the rfft of a length-(L+H) series; irfft gives a
    direct time-domain extrapolation, whose last H values are the forecast."""
    def __init__(self, L, H):
        super().__init__()
        self.L, self.H = L, H
        n_in = L // 2 + 1
        n_out = (L + H) // 2 + 1
        self.weight = nn.Parameter(torch.randn(n_out, n_in, dtype=torch.cfloat) * 0.02)

    def forward(self, x):  # x: (B, L) real
        xf = torch.fft.rfft(x.float(), n=self.L, dim=-1)    # (B,n_in) complex
        yf = torch.einsum("oi,bi->bo", self.weight, xf)      # (B,n_out) complex
        y = torch.fft.irfft(yf, n=self.L + self.H, dim=-1)   # (B,L+H) real
        return y[:, -self.H:]


class SpectraMix(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=64, patch_len=24, layers=2, d_ff=128, dropout=0.1, dh=64, **kw):
        super().__init__()
        self.L, self.H, self.patch_len = L, H, patch_len
        self.n_patches = L // patch_len
        self.n_time, self.n_weather, self.use_weather = n_time, n_weather, use_weather
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.blocks = nn.ModuleList([_MixerBlock(self.n_patches, d_model, d_ff, dropout) for _ in range(layers)])
        self.mix_head = nn.Linear(self.n_patches * d_model, H)
        self.scale_head = nn.Sequential(nn.LayerNorm(self.n_patches * d_model), nn.Linear(self.n_patches * d_model, H))
        self.freq = FrequencyExtrapolation(L, H)
        self.freq_gate = nn.Parameter(torch.tensor(0.0))    # sigmoid-gated blend weight, starts at 0.5
        if use_weather:
            self.Tbal_heat = nn.Parameter(torch.tensor(-0.3))
            self.Tbal_cool = nn.Parameter(torch.tensor(0.3))
            self.dd_mlp = nn.Sequential(nn.Linear(2, dh), nn.GELU(), nn.Linear(dh, 1))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        patches = self.patch_embed(yn.view(B, self.n_patches, self.patch_len))
        z = patches
        for blk in self.blocks:
            z = blk(z)
        flat = z.reshape(B, -1)
        mu_mix = self.mix_head(flat)
        mu_freq = self.freq(yn)
        mu_n = mu_mix + torch.sigmoid(self.freq_gate) * mu_freq
        raw_scale = self.scale_head(flat)

        if self.use_weather:
            _, weather, _ = split_exo(exo, self.n_time, self.n_weather, True)
            temp_fut = weather[:, self.L:, 0]      # channel 0 = confirmed dry-bulb temperature
            heat = F.relu(self.Tbal_heat - temp_fut)
            cool = F.relu(temp_fut - self.Tbal_cool)
            mu_n = mu_n + self.dd_mlp(torch.stack([heat, cool], -1)).squeeze(-1)
        return mu_n, raw_scale, mu, sd


# ===========================================================================
# 14) MambaPatch -- selective state-space scan (Gu & Dao 2023, "Mamba:
#     Linear-Time Sequence Modeling with Selective State Spaces", the S6
#     mechanism) over daily patches, pure PyTorch (no custom CUDA kernel --
#     the official mamba-ssm package's compiled kernel is often unreliable to
#     install in Colab). The previous draft of this study explicitly
#     excluded Mamba/selective-SSM as "too many nuances/bugs for this
#     context" -- this fills that gap with a small, sequential (7-step,
#     patch-level) scan, cheap enough not to need a fused kernel.
# ===========================================================================
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=8):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.5 + 0.5))  # A=-exp(A_log) in [-1,-0.5]
        self.D = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

    def forward(self, x):  # x: (B, T, d_model)
        Bsz, T, D = x.shape
        A = -torch.exp(self.A_log)                                              # (D,d_state), strictly negative
        delta = F.softplus(self.delta_proj(x))                                   # (B,T,D)
        Bt = self.B_proj(x)                                                       # (B,T,d_state)
        Ct = self.C_proj(x)                                                       # (B,T,d_state)

        h = x.new_zeros(Bsz, D, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(delta[:, t].unsqueeze(-1) * A.unsqueeze(0))                    # (B,D,d_state)
            dBx = (delta[:, t] * x[:, t]).unsqueeze(-1) * Bt[:, t].unsqueeze(1)             # (B,D,d_state)
            h = dA * h + dBx
            ys.append((h * Ct[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t])               # (B,D)
        return torch.stack(ys, 1)                                                            # (B,T,D)


class MambaPatch(nn.Module):
    def __init__(self, L=168, H=24, n_time=3, n_weather=7, use_weather=True,
                 d_model=96, d_state=8, patch_len=24, layers=2, dropout=0.1, dh=128, **kw):
        super().__init__()
        self.L, self.H, self.patch_len = L, H, patch_len
        self.n_patches = L // patch_len
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.in_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(layers)])
        self.gate_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(layers)])
        self.ssms = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        c_exo = c_exo_width(n_time, n_weather, use_weather)
        self.fut = nn.Sequential(nn.Linear(H * c_exo, dh), nn.GELU(), nn.Linear(dh, d_model))
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 2 * H))

    def forward(self, yh, exo):
        B = yh.size(0)
        yn, mu, sd = self.revin(yh)
        z = self.patch_embed(yn.view(B, self.n_patches, self.patch_len))        # (B,7,d_model)

        for inp, gate, ssm, outp, norm in zip(self.in_proj, self.gate_proj, self.ssms, self.out_proj, self.norms):
            residual = z
            x = F.silu(inp(norm(z)))
            y = ssm(x)
            y = y * F.silu(gate(z))
            z = residual + self.dropout(outp(y))
        summary = z.mean(1)                                                       # (B,d_model)

        fut = self.fut(exo[:, self.L:].reshape(B, -1))
        out = self.head(torch.cat([summary, fut], -1)).view(B, self.H, 2)
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
    "transformer_s": TransformerGaussian,
    "transformer_m": TransformerGaussian,
    "transformer_l": TransformerGaussian,
    "tftlite": TFTLite,
    "xlstm": XLSTMPatch,
    "spectramix": SpectraMix,
    "mamba": MambaPatch,
}

MODEL_KW = {
    "persistence_avg": dict(),
    "persistence_last_day": dict(),
    "persistence_last_week": dict(),
    "lstm": dict(hidden=96, layers=2, dropout=0.1),
    "gru": dict(hidden=96, layers=2, dropout=0.1),
    "dlinear": dict(kernel=25, dh=128, dropout=0.1),
    # Transformer baselines are sized to a >=7M-parameter floor (both weather
    # conditions) so capacity is never the reason a canonical architecture
    # loses to persistence -- verified by instantiation, see MIN_TRANSFORMER_PARAMS.
    "patchtst": dict(patch_len=16, stride=8, d_model=384, e_layers=6, n_heads=16, d_ff=768, dropout=0.1),
    "itransformer": dict(d_model=512, e_layers=4, n_heads=8, d_ff=1024, dropout=0.1),
    "timexer": dict(d_model=448, n_heads=8, layers=4, d_ff=896, patch_len=24, stride=12, dropout=0.1),
    "informer": dict(d_model=384, n_heads=8, e_layers=5, d_ff=768, factor=5, dropout=0.1),
    "autoformer": dict(d_model=384, n_heads=8, e_layers=4, d_ff=768, kernel=25, dropout=0.1),
    "crossformer": dict(d_model=320, n_heads=8, layers=4, seg_len=24, n_routers=8, dropout=0.15),
    # Exact hyperparameters from buildings_bench/configs/TransformerWithGaussian-{S,M,L}.toml
    "transformer_s": dict(d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                          dim_feedforward=512, dropout=0.0),
    "transformer_m": dict(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                          dim_feedforward=1024, dropout=0.0),
    "transformer_l": dict(d_model=768, nhead=12, num_encoder_layers=12, num_decoder_layers=12,
                          dim_feedforward=2048, dropout=0.0),
    # Novel research-grounded architectures (see study.ipynb intro for citations).
    # tftlite is attention-based, so it follows the same >=7M floor as the
    # canonical transformers; the other three are deliberately compact
    # (their parameter efficiency is part of what they test).
    "tftlite": dict(d_model=320, n_heads=8, dropout=0.1),
    "xlstm": dict(d_model=96, patch_len=24, layers=2, dropout=0.1, dh=128),
    "spectramix": dict(d_model=64, patch_len=24, layers=2, d_ff=128, dropout=0.1, dh=64),
    "mamba": dict(d_model=96, d_state=8, patch_len=24, layers=2, dropout=0.1, dh=128),
}

# Capacity floor for the transformer-family baselines above -- asserted by the
# smoke test in study.ipynb's training markdown and checkable via:
#   all(count_params(build(n, use_weather=uw, **MODEL_KW[n])) >= MIN_TRANSFORMER_PARAMS ...)
MIN_TRANSFORMER_PARAMS = 7_000_000
TRANSFORMER_FAMILY = ["patchtst", "itransformer", "timexer", "informer",
                      "autoformer", "crossformer", "tftlite"]


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
