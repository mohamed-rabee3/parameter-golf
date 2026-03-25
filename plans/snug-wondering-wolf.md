# Parameter Golf: Implementation Plan to Beat 1.1194 BPB

## Context

The current SOTA on OpenAI's Parameter Golf (16MB / 10-min track) is **1.1194 BPB** using an 11-layer 512d transformer with LeakyReLU(0.5)^2, legal TTT, Parallel Muon, GPTQ-lite int6, and EMA+SWA. The goal is to push below 1.119 BPB toward ~1.10-1.11 range through architecture, training, and quantization improvements — **without offline distillation**.

**Base file:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` (the current record script, ~1900 lines). All work builds on this file incrementally.

**Key constraint:** artifact <= 16,000,000 bytes, training <= 600s on 8xH100, reproducible across 3+ seeds.

---

## Phase 1: Quick Wins (estimated -0.003 to -0.008 BPB cumulative)

### 1.1 Selective TTT (estimated -0.002 BPB)

**Why:** Current TTT updates all blocks (SGD on everything). E2E TTT literature shows updating only MLPs is more stable, allowing higher LR and more epochs for the same compute.

**File:** `train_gpt.py`, function `eval_val_sliding_ttt()` (lines ~1074-1229)

**Changes:**
1. In the TTT parameter collection (lines ~1110-1123), change freeze logic:
   ```python
   # CURRENT: freezes first N blocks entirely
   # NEW: freeze all attention weights + embeddings + LN, update only MLP banks
   ttt_params = []
   for name, p in base_model.named_parameters():
       p.requires_grad = False  # freeze everything first
   # Only unfreeze MLP bank weights
   base_model.mlp_up_bank.requires_grad_(True)
   base_model.mlp_down_bank.requires_grad_(True)
   ttt_params = [base_model.mlp_up_bank, base_model.mlp_down_bank]
   ```
2. Increase TTT LR from 0.002 to 0.008 (safe because only MLPs are updating)
3. Increase TTT epochs from 3 to 5
4. Add per-bank LR scaling: higher for middle layers, lower for top/bottom:
   ```python
   # Use separate param groups with different LR multipliers
   # Middle layers (4-7) get 1.5x LR, edge layers get 0.7x LR
   ```

**Ablation:** Compare post-TTT BPB for:
- Baseline (all blocks, lr=0.002, 3 epochs) = -0.0025 BPB gain
- MLP-only (lr=0.005, 3 epochs)
- MLP-only (lr=0.008, 5 epochs)
- MLP-only with per-layer LR

**Risk:** Low. TTT is post-training; if it regresses, revert to baseline TTT.

---

### 1.2 LeakyReLU Slope Sweep (estimated -0.001 BPB)

**Why:** The 0.5 slope was a single-shot finding. Different slopes may quantize better at int6.

**File:** `train_gpt.py`, MLP class (lines ~724-730)

**Changes:**
1. Make slope configurable via env var:
   ```python
   leaky_slope = float(os.environ.get("LEAKY_SLOPE", "0.5"))
   ```
2. In MLP.forward:
   ```python
   x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=leaky_slope)
   ```

**Ablation:** Run 1-seed sweeps at slopes {0.2, 0.3, 0.4, 0.5, 0.6}. Measure both pre-TTT and post-quant BPB (slope affects quantization friendliness).

**Risk:** Negligible. One-line change, easy to revert.

---

### 1.3 Earlier QAT (estimated -0.001 to -0.002 BPB)

**Why:** Current late QAT at threshold 0.15 gives weights only ~15% of training to adapt to quantization noise. EfficientQAT shows starting earlier improves post-quant quality.

**File:** `train_gpt.py`, line ~1676

**Changes:**
1. Change `LATE_QAT_THRESHOLD` default from 0.15 to 0.25
2. Add a two-phase QAT option:
   ```python
   # Phase 1: Block-wise QAT (threshold 0.30) - freeze non-current blocks
   # Phase 2: End-to-end QAT (threshold 0.15) - all blocks active
   ```
3. Reduce LR by 0.5x when QAT activates:
   ```python
   if CastedLinear._qat_enabled:
       effective_lr *= 0.5
   ```

**Ablation:** Compare thresholds {0.15, 0.20, 0.25, 0.30} on post-quant BPB.

**Risk:** Low. If earlier QAT hurts training loss, the warmdown schedule self-corrects.

---

### 1.4 4x MLP Width (estimated -0.004 to -0.008 BPB)

**Why:** Ternary submission ablated this: 4x MLP gives -0.008 BPB over 3x. MLP is where most model capacity lives.

**File:** `train_gpt.py`, model config and bank creation

**Changes:**
1. Change `MLP_MULT` from 3 to 4:
   ```python
   mlp_mult = float(os.environ.get("MLP_MULT", "4"))
   ```
2. This increases `mlp_up_bank` from (11, 1536, 512) to (11, 2048, 512) and `mlp_down_bank` similarly.
3. **Budget check:** At int6, the MLP banks go from ~17.7M params to ~23.6M params. At 6 bits each + LZMA overhead, this adds ~4.4MB. Current artifact is 15.95MB. **This does NOT fit at int6 alone.**

**Resolution options (pick one):**
- **(a)** Combine with layerwise bit allocation (Phase 2.1) — int5 for layers 0-3 MLP saves ~1.5MB
- **(b)** Reduce to 3.5x MLP (compromise): adds ~2.2MB, fits within budget
- **(c)** Reduce BigramHash from 1536 to 1024 buckets, saving ~0.3MB, and reduce VE layers
- **(d)** Combine with 8192 BPE + factored embeddings (Phase 2.2) which saves ~4MB on embeddings

**Recommendation:** Implement 4x MLP together with Phase 2.1 (layerwise bits) or Phase 2.2 (factored embeddings). For Phase 1 ablation, test 3.5x first as a safe intermediate.

**Risk:** Medium. Requires careful budget management. May need to defer to Phase 2.

---

## Phase 2: Medium Engineering (estimated -0.005 to -0.010 BPB cumulative)

### 2.1 Layerwise Bit Allocation (estimated -0.001 to -0.003 BPB)

**Why:** Early layers are less sensitive to quantization noise than top layers. Asymmetric bit allocation maximizes effective capacity under fixed budget.

**File:** `train_gpt.py`, functions `quantize_int6_per_row()` (lines ~1242-1261) and `mixed_quantize_int6()` (lines ~1330-1359)

**Changes:**
1. Add per-layer quantization config:
   ```python
   # Layer bit allocation: {layer_idx: bits}
   LAYER_BITS = {
       0: 5, 1: 5, 2: 5, 3: 5,     # early layers: int5
       4: 6, 5: 6, 6: 6, 7: 6,     # middle layers: int6
       8: 7, 9: 7, 10: 7,           # top layers: int7
   }
   ```
2. Modify `quantize_int_per_row()` to accept variable clip range:
   ```python
   def quantize_int_per_row(t, clip_range):
       # clip_range = 15 for int5, 31 for int6, 63 for int7
   ```
3. In `mixed_quantize_int6()`, detect layer index from parameter name and apply appropriate bit width
4. Update fake QAT in `CastedLinear` to use per-layer bit width during training:
   ```python
   # During forward, look up which layer this weight belongs to
   # Apply appropriate clip_range for STE
   ```

**Budget impact:** int5 layers save ~17% per layer vs int6. int7 layers cost ~17% more. Net: roughly neutral if balanced, or slight savings if more layers are int5.

**Ablation:** Compare uniform int6 vs layerwise {int5/int6/int7} on post-quant BPB.

**Risk:** Medium. Requires modifying the quantization pipeline and QAT STE. Need to verify LZMA compression ratios don't change adversely with mixed bit widths.

---

### 2.2 8192 BPE + Factored Embeddings (estimated -0.003 to -0.005 BPB)

**Why:** Larger vocab captures more semantic content per token, improving tokens-per-byte ratio. Factored embeddings (8192x254 bottleneck) save ~4MB vs full-dim, enabling wider MLP.

**File:** `train_gpt.py`, GPT class embedding setup and forward pass

**Changes:**
1. Add factored embedding support:
   ```python
   embed_dim = int(os.environ.get("EMBED_DIM", "0"))  # 0 = full dim
   # In GPT.__init__:
   self.embed_dim = embed_dim if embed_dim > 0 else model_dim
   self.tok_emb = nn.Embedding(vocab_size, self.embed_dim)
   self.embed_proj = CastedLinear(self.embed_dim, model_dim, bias=False) if self.embed_dim != model_dim else None
   self.embed_proj_rev = CastedLinear(model_dim, self.embed_dim, bias=False) if self.embed_dim != model_dim else None
   ```
2. In forward pass:
   ```python
   x = self.tok_emb(input_ids)
   if self.embed_proj is not None:
       x = self.embed_proj(x)
   ```
3. In logit computation (tied embeddings):
   ```python
   if self.embed_proj_rev is not None:
       x_proj = self.embed_proj_rev(x_flat)
       logits_proj = F.linear(x_proj, self.tok_emb.weight)
   else:
       logits_proj = F.linear(x_flat, self.tok_emb.weight)
   ```
4. Update data loading to use 8192 BPE dataset and tokenizer:
   ```
   DATA_PATH=./data/datasets/fineweb10B_sp8192
   TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
   VOCAB_SIZE=8192
   EMBED_DIM=254
   ```
5. Update BPB calculation - the tokenizer-agnostic BPB formula already handles different vocab sizes correctly (lines ~180-204 build per-token byte LUTs from the tokenizer).

**Budget impact:** 8192x254 embedding = 2.08M params (vs 1024x512 = 0.52M). But at int8 with LZMA, this is ~2MB. The factored projection layers add ~130K params each. Net: +~1.5MB for embeddings, but saves ~4MB if we were using full-dim 8192x512. Combined with 4x MLP, fits in 16MB.

**Ablation:** Compare {sp1024 full-dim} vs {sp8192 factored-254} on BPB. Must verify BPB calculation is correct with new tokenizer.

**Risk:** Medium-High. Changing tokenizer requires careful BPB validation. Must download sp8192 dataset. BigramHash constants may need re-tuning for 8192 vocab.

---

### 2.3 YaRN Positional Encoding (estimated -0.001 to -0.002 BPB)

**Why:** YaRN enables training at seq_len=1024 but effective context of 2048 during eval. Both ternary and binary submissions use YaRN over partial RoPE.

**File:** `train_gpt.py`, Rotary class

**Changes:**
1. Port YaRN from ternary script (lines 565-597):
   ```python
   rope_type = os.environ.get("ROPE_TYPE", "rope")  # "rope" or "yarn"
   yarn_max_len = int(os.environ.get("YARN_MAX_LEN", "2048"))

   # In Rotary.__init__:
   if rope_type == "yarn":
       scale = train_seq_len / yarn_max_len
       freq_idx = torch.arange(0, dim, 2, dtype=torch.float32)
       ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
       inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
   ```
2. Replace `ROPE_DIMS=16` (partial RoPE) with full-dim YaRN
3. Set `ROPE_BASE=5000` (from ternary submission, tuned for YaRN)

**Ablation:** Compare partial RoPE (16/64) vs YaRN (full dim, max_len=2048) at stride-64 and stride-16.

**Risk:** Low. Drop-in replacement for Rotary class. May need to adjust eval stride.

---

### 2.4 Stride-16 Sliding Eval (estimated -0.001 to -0.003 BPB)

**Why:** Ternary submission shows stride-16 gives -0.025 BPB over chunked eval. Current SOTA uses stride-64.

**File:** `train_gpt.py`, eval functions and TTT

**Changes:**
1. Change `EVAL_STRIDE` from 64 to 16
2. This increases eval time ~4x. Current eval takes ~120s at stride-64, so stride-16 would take ~480s.
3. **Time budget concern:** Training (600s) + eval (480s) + TTT (410s) = 1490s total. This is fine — the 10-min limit only applies to training, not evaluation.
4. May need to increase `SLIDING_BATCH_SIZE` for throughput if memory allows.

**Ablation:** Compare stride {16, 32, 64} on BPB. Verify timing fits.

**Risk:** Low. Only affects evaluation, not training or artifact.

---

### 2.5 SmearGate Integration (estimated -0.002 to -0.004 BPB net)

**Why:** SmearGate gives -0.007 BPB in binary submission but costs 22ms/step (~264 fewer steps in 600s). Need to ablate whether BPB gain > step-count loss.

**File:** `train_gpt.py`, Block class and GPT class

**Changes:**
1. Port SmearModule from ternary/binary scripts:
   ```python
   class SmearModule(nn.Module):
       def __init__(self, dim):
           super().__init__()
           self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
       def forward(self, x):
           cumsum = x.cumsum(dim=1)
           counts = torch.arange(1, x.size(1)+1, device=x.device, dtype=x.dtype).view(1,-1,1)
           smeared = cumsum / counts
           gate = torch.tanh(self.gate.to(dtype=x.dtype))
           return x + gate * (smeared - x)
   ```
2. Add to GPT after initial embedding + RMSNorm (before encoder blocks):
   ```python
   smear_enabled = bool(int(os.environ.get("SMEAR", "0")))
   if smear_enabled:
       self.smear = SmearModule(model_dim)
   ```
3. Add `self.gate` to scalar params (AdamW, not Muon)

**Ablation:** Measure ms/step increase, total steps completed, and net BPB change. The break-even is: if SmearGate costs N fewer steps, it must give > N * (BPB_improvement_per_step) in BPB.

**Risk:** Medium. The 22ms/step overhead may not be worth it at the 10-min boundary. Strongly dependent on step time — if FlashAttention-3 integration (if not already used) offsets the cost, SmearGate becomes clearly positive.

---

### 2.6 Poly5 Softcap (estimated -0.0005 BPB)

**Why:** Both ternary and binary submissions use Poly5 (cap=10) instead of tanh (cap=30). Polynomial approximation is faster and the lower cap may regularize better.

**File:** `train_gpt.py`, logit computation in GPT.forward

**Changes:**
1. Add softcap type config:
   ```python
   softcap_type = os.environ.get("SOFTCAP_TYPE", "tanh")
   ```
2. Implement Poly5:
   ```python
   def softcap(self, logits):
       s = self.logit_softcap
       if self.softcap_type == "poly":
           x_sc = torch.clamp(logits / s, -2.0, 2.0)
           x2 = x_sc * x_sc
           return s * torch.clamp(x_sc * (1.0 - x2/3.0 + x2*x2/15.0), -1.0, 1.0)
       return s * torch.tanh(logits / s)
   ```

**Ablation:** Compare tanh(cap=30) vs poly5(cap=10) vs poly5(cap=30).

**Risk:** Very low. Simple function swap.

---

## Phase 3: High-Risk/High-Reward (estimated -0.005 to -0.015 BPB cumulative)

### 3.1 Ternary MLP Hybrid Architecture (estimated -0.005 to -0.010 BPB)

**Why:** Ternary MLPs use 1.6 bits/param instead of 6, allowing 3-4x more parameters. The ternary submission gets 1.157 BPB without TTT at 73.7M params. A hybrid (ternary MLPs + int6 attention) at 768d with TTT could break 1.11.

**File:** New file `train_gpt_hybrid.py` based on SOTA script

**Changes:**
1. **Add TernaryLinear class** (port from ternary script lines 452-465):
   ```python
   class TernaryLinear(nn.Module):
       def __init__(self, in_features, out_features, group_size=128):
           super().__init__()
           self.weight = nn.Parameter(torch.empty(out_features, in_features))
           self.group_size = group_size
       def forward(self, x):
           w = self.weight.bfloat16()
           w_g = w.reshape(-1, self.group_size)
           scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
           q = (w_g / scale).round().clamp(-1, 1)
           w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
           return F.linear(x, w_ternary)
   ```

2. **Replace MLP banks with ternary banks:**
   ```python
   # Instead of CastedLinear (FP32 stored, BF16 computed):
   # MLP weights: ternary (1.6 bits/param)
   self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
   self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
   # These get ternary quantization in forward pass
   ```

3. **Keep attention in int6/int8:**
   ```python
   # qo_bank, kv_bank remain as FP32/BF16, quantized to int6 for artifact
   self.qo_bank = nn.Parameter(torch.empty(2*num_layers, model_dim, model_dim))
   self.kv_bank = nn.Parameter(torch.empty(2*num_layers, kv_dim, model_dim))
   ```

4. **Scale to 768d, 11-12 layers:**
   - MLP 4x: up_bank = (12, 3072, 768), down_bank = (12, 768, 3072)
   - At ternary: 12 * 3072 * 768 * 2 * 1.6 bits = ~9MB for MLP
   - Attention at int6: 24 * 768 * 768 * 6 bits + 24 * 256 * 768 * 6 bits = ~10.5MB ... too much
   - **Need to use int5 for attention or reduce to 10-11 layers**
   - Alternative: 768d, 10 layers, 4x MLP ternary + int6 attention:
     - MLP ternary: ~7.5MB
     - Attention int6: ~6.7MB
     - Embeddings + scalars + code: ~1.5MB
     - Total: ~15.7MB -- fits!

5. **Compression pipeline:**
   ```python
   # For ternary MLP weights: base-3 packing + LZMA (from ternary script)
   # For int6 attention weights: existing GPTQ-lite + LZMA
   # For embeddings/scalars: FP8 or int8 + LZMA
   ```

6. **Optimizer:**
   - Muon for all 2D weight banks (both ternary MLP and int6 attention)
   - NeoMuon (3 NS steps) may help with ternary STE gradient attenuation
   - AdamW for embeddings and scalars

**Budget calculation for target config (768d, 10L, 4x MLP):**
| Component | Params | Bits | Compressed Est. |
|-----------|--------|------|-----------------|
| MLP ternary (up+down) | 47.2M | 1.6 | ~7.0MB |
| Attn int6 (QO+KV) | 14.7M | 6 | ~6.5MB |
| Embed (8192x254) + proj | 2.5M | 8 | ~1.5MB |
| Scalars, LN, VE, etc. | 0.3M | 32 | ~0.5MB |
| Code | - | - | ~0.07MB |
| **Total** | **~65M** | - | **~15.6MB** |

**Ablation strategy:**
1. First: port ternary MLP to 512d/11L (same as SOTA) and compare pre-TTT BPB
2. Then: scale to 768d/10L and compare
3. Then: add TTT and compare full pipeline

**Risk:** High. Ternary training converges slower (the submission shows 1.157 at 10min vs 1.122 pre-TTT for int6). The question is whether TTT can close the gap + the wider model compensates. The binary submission reaching 1.1239 at 2h (no TTT) suggests the capacity is there — the challenge is convergence speed.

---

### 3.2 TTT on Hybrid Model (estimated -0.003 to -0.005 BPB)

**Why:** Neither the ternary nor binary submissions used TTT. Adding selective MLP-only TTT to the hybrid should give similar or better gains than on the int6 model, since the ternary MLPs have more parameters to adapt.

**File:** `train_gpt_hybrid.py`, TTT function

**Changes:**
1. TTT updates only the ternary MLP banks (which have more capacity)
2. During TTT, use full-precision weights (not ternary-quantized) for gradient computation
3. After TTT, re-quantize to ternary for the next chunk's forward pass
4. Higher TTT LR (0.01-0.02) since ternary weights have larger effective learning rate
5. Consider using Adam instead of SGD for TTT (better for sparse ternary gradients)

**Risk:** Medium. Ternary weights may be harder to fine-tune via gradient descent. The STE noise during TTT could cause instability. Mitigation: use full-precision copies during TTT, only quantize for inference.

---

### 3.3 Dynamic TTT Epochs + Temperature Scaling (estimated -0.001 BPB)

**File:** `train_gpt.py` or `train_gpt_hybrid.py`, TTT function

**Changes:**
1. **Dynamic epochs:** Track loss improvement per epoch; stop early if gain < threshold:
   ```python
   for epoch in range(max_ttt_epochs):
       epoch_loss = train_one_epoch(chunk)
       if epoch > 0 and prev_loss - epoch_loss < min_improvement:
           break
       prev_loss = epoch_loss
   ```
2. **Temperature scaling** (port from ternary script):
   ```python
   def find_temp(model, cal_data):
       for t in [0.80, 0.85, 0.90, 0.95, 1.00]:
           loss = eval_with_temp(model, cal_data, t)
           if loss < best: best_t = t
       return best_t
   ```
3. Apply temperature during sliding eval (after TTT)

**Risk:** Low. Both are post-training modifications.

---

## Implementation Order & Dependencies

```
Week 1: Phase 1 (all independent, can run ablations in parallel)
  Day 1-2: 1.2 LeakyReLU sweep (trivial, immediate ablation data)
  Day 1-2: 1.3 Earlier QAT sweep (trivial, immediate ablation data)
  Day 2-3: 1.1 Selective TTT (modify TTT function, run ablations)
  Day 3-4: 1.4 MLP width experiments (3.5x at int6, check budget)
  Day 4-5: Combine best Phase 1 results into single config

Week 2: Phase 2 (some dependencies)
  Day 1-2: 2.1 Layerwise bit allocation (enables 4x MLP)
  Day 1-2: 2.3 YaRN (independent, drop-in)
  Day 2-3: 2.6 Poly5 softcap (independent, trivial)
  Day 2-4: 2.2 8192 BPE + factored embeddings (larger change, needs data download)
  Day 3-4: 2.4 Stride-16 eval (depends on YaRN for best results)
  Day 4-5: 2.5 SmearGate (independent, but ablate carefully vs step count)
  Day 5: Combine best Phase 2 results with Phase 1 winner

Week 3-4: Phase 3
  Day 1-3: 3.1 Build ternary hybrid (largest engineering effort)
  Day 3-4: Port all Phase 1+2 wins to hybrid
  Day 4-5: 3.2 TTT on hybrid
  Day 5-6: 3.3 Dynamic TTT + temperature scaling
  Day 6-7: Final tuning, 3-seed validation runs
```

## Verification Plan

For each change:
1. **Single-seed quick run** (seed=1337): check pre-TTT BPB, post-quant BPB, artifact size
2. **Budget check:** verify artifact <= 16,000,000 bytes
3. **Timing check:** verify training completes in <= 600s
4. **If promising, 3-seed run** (seeds 1337, 42, 2025): check mean and std
5. **Record threshold:** improvement must be >= 0.005 nats with p<0.01 across 3 seeds

**Key metrics to track per run:**
- `pre_ttt_bpb` (after training, before TTT)
- `post_ttt_bpb` (after TTT)
- `rt_bpb` (roundtrip: after quantize + decompress)
- `artifact_bytes`
- `ms_per_step`
- `total_steps`
- `training_wall_seconds`
- `ttt_wall_seconds`

## Critical Files to Modify

| File | Purpose |
|------|---------|
| `records/.../2026-03-23_.../train_gpt.py` | Base SOTA script (Phases 1-2) |
| `train_gpt_hybrid.py` (new) | Hybrid ternary/int6 script (Phase 3) |
| `records/.../ternary/train_gpt_cuda_ternary.py` | Reference for ternary code to port |
| `records/.../binary/.../train_gpt_cuda_binary.py` | Reference for SmearGate, YaRN |

## Existing Code to Reuse (not rewrite)

- **TernaryLinear**: port directly from ternary script lines 452-465
- **SmearModule**: port directly from ternary/binary scripts lines 695-705
- **YaRN Rotary**: port directly from ternary script lines 565-597
- **Poly5 softcap**: port directly from ternary script lines 859-865
- **Base-3 packing**: port directly from ternary script lines 114-138
- **FP8 QAT**: port directly from ternary script lines 418-431
- **Temperature scaling**: port directly from ternary script lines 1041-1055
- **Parameter Banking + Parallel Muon**: already in SOTA script, reuse as-is
- **GPTQ-lite**: already in SOTA script, extend for layerwise bits
- **TTT framework**: already in SOTA script, modify freeze/LR logic
