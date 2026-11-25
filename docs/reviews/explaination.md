# Theory behind the SD fusion adapter, its initialisation, gradient clipping and EMA

This note explains, from a theoretical point of view, the design choices listed in
[param_sd_dinoLarge.md](param_sd_dinoLarge.md) for the presets `unified_lora_large_sd` and
`unified_lora_large_sd_mix` (decisions D-009 and D-011 in [decisions.md](decisions.md)). Code references:
`src/models/fusion.py`, `src/models/sd.py`, `src/utils/ema.py`, `src/train.py`.

Notation, per feature cell (one of the 46 x 46 grid positions):

- `d` = DINOv2-L feature, 1024-d, L2-normalised (`||d|| = 1`).
- `z` = Stable Diffusion feature, 3200-d (up-blocks 0, 1, 2 = 1280 + 1280 + 640 channels, each block
  L2-normalised, concatenated, renormalised).
- `s = out(blocks(inp(Dropout2d(z))))` = the adapter output, 1024-d.
- `f` = fused feature used for matching: `f = normalize(d + s)` ("adapter") or `f = normalize(W_d d + W_s s + b)`
  ("adapter_mix").

Matching uses cosine similarity of `f` between the source point and every target cell, followed by a softmax
with temperature T. Every quantity below is about how `f` changes during training.

---

## 1. The adapter architecture

```
z (3200) -> inp: 1x1 conv 3200 -> 1024
         -> block 1: h + conv1x1(1024->256) . GELU . conv3x3(256->256) . GELU . conv1x1(256->1024)
         -> block 2: same
         -> out: 1x1 conv 1024 -> 1024   (weight = 0, bias = 0 at init)
         = s
```

Parameter counts (checked by hand):

| Layer | Formula | Parameters |
|---|---|---:|
| `inp` | 3200 x 1024 + 1024 | 3,277,824 |
| bottleneck 1x1 down | 1024 x 256 + 256 | 262,400 |
| bottleneck 3x3 | 256 x 256 x 9 + 256 | 590,080 |
| bottleneck 1x1 up | 256 x 1024 + 1024 | 263,168 |
| one bottleneck block | sum of the three | 1,115,648 |
| `blocks` (2 blocks) | 2 x 1,115,648 | 2,231,296 |
| `out` | 1024 x 1024 + 1024 | 1,049,600 |

### 1.1 Why `inp` is a 1x1 convolution

A 1x1 convolution is a linear map applied to each cell independently: `inp(z) = W_in z + b_in`. It does two
things:

1. **Dimension change.** SD features (3200-d) and DINOv2 features (1024-d) live in different spaces with
   different sizes. To add them (`d + s`) they must have the same size, so `inp` projects SD into 1024 channels.
2. **Feature selection.** SD up-block features are known to mix semantic information (deep blocks, low
   resolution) with appearance and texture information (shallow blocks, high resolution). A learned linear
   projection can weight the channels that help correspondence and suppress the others. It is the learned
   version of the fixed `sd_concat_weight` used in "concat".

It is per cell, so it does not move information between positions and cannot blur the location of a keypoint.

### 1.2 Why bottleneck blocks (1024 -> 256 -> 256 -> 1024)

The bottleneck is the ResNet-50 design (He et al., 2016). Its purpose is to add **spatial context** and
**non-linearity** at a low cost.

**Cost argument.** The only layer that looks at neighbouring cells is the 3x3 convolution. A 3x3 convolution at
full width (1024 -> 1024) would cost `1024 x 1024 x 9 = 9.4M` parameters per layer. Squeezing to 256 channels
first makes it `256 x 256 x 9 = 0.59M`, 16 times less. The two 1x1 layers around it (0.26M each) only change the
width. A whole block costs 1.1M instead of 9.4M for a single wide 3x3.

**Capacity argument.** The 256-d middle is a low-rank constraint: the residual update `body(h)` of each block goes
through a 256-d space. With about 470 optimiser updates (6 epochs x ~78) and a few thousand training pairs, a
smaller hypothesis space overfits less. This is the same idea as LoRA (a low-rank update), applied to the SD branch.

**Non-linearity.** Without the GELUs, `inp`, `blocks` and `out` would collapse into a single linear map
(`W_out W_blocks W_in z`). The GELUs let the adapter compute non-linear functions of the SD features, for example
"keep channel k only where channel j is active". This is why the adapter is more than a projection.

**Residual connection.** Each block computes `h + body(h)`. The identity path keeps the projected SD signal
`inp(z)` intact and lets the gradient reach `inp` without passing through the GELUs. A block can learn
`body = 0` and become an identity, so adding blocks never makes the starting function worse.

### 1.3 Implications of the bottleneck

- **Receptive field.** Each 3x3 convolution sees 3 x 3 cells. Two blocks give a 5 x 5 cell receptive field, about
  70 x 70 pixels at the 644 input (14 px per cell). The adapter can therefore:
  - smooth SD features, which are noisier than DINOv2 features (they come from a single noisy diffusion step);
  - add local context ("this cell is next to a wheel").
  The cost is a possible loss of sharpness: if the 3x3 layers average neighbours too much, the similarity peak
  becomes wider, and the soft-argmax becomes less precise. Small objects (tight PCK tolerance) suffer most. The
  DINOv2 path is untouched, so the final peak still has the DINOv2 sharpness; the SD part should add semantics,
  not localisation.
- **Padding.** `padding=1` adds zeros outside the map. Cells at the border see zeros as neighbours, so the adapter
  output is slightly different there. With 46 cells per side this affects only the outer ring.
- **No normalisation layer** (no BatchNorm, no LayerNorm). The input `z` is already L2-normalised per cell, so
  its scale is fixed. With pair batches of 4 images, BatchNorm statistics would be unreliable anyway. The
  consequence is that the scale of `s` is controlled only by the weights (and by weight decay).
- **Size relative to LoRA.** The adapter has 6.56M trained parameters at `lr_adapter` 3e-4. The LoRA group has
  0.91M at `lr_lora` 8e-5. The SD branch is the larger and faster-moving part of the model, so it is the more
  likely source of overfitting. `adapter_dropout` 0.2 (`Dropout2d`) answers this: it drops whole SD channels
  (the same channel in every cell) during training, so the adapter cannot rely on a few channels.

### 1.4 Why the norm of `s` is an implicit gate

`f = normalize(d + s)` with `||d|| = 1`. The cosine similarity between two fused features is

```
cos(f1, f2) = (d1.d2 + d1.s2 + s1.d2 + s1.s2) / (||d1 + s1|| ||d2 + s2||)
```

If `||s|| << 1`, the DINOv2 term `d1.d2` dominates and the model behaves like DINOv2. If `||s||` is around 1 or
more, the SD terms count as much as DINOv2. So there is no explicit mixing weight: **the magnitude of `s`,
learned by `out`, decides how much SD counts.** Weight decay (0.05) pulls the weights of `out` toward zero, so it
biases the model toward DINOv2 unless SD lowers the loss. "adapter_mix" makes this gate explicit and
per-channel through `W_d` and `W_s` (section 3).

---

## 2. Zero initialisation of `out`

`out.weight = 0` and `out.bias = 0` give `s = 0` at update 0, so `f = normalize(d) = d`: the model starts
exactly from the DINOv2-L LoRA model (which itself starts from frozen DINOv2-L, because `lora_B = 0`).

### 2.1 Why start at zero

A randomly initialised adapter would add a random 1024-d vector to every DINOv2 feature. With `inp` and the
blocks at PyTorch's default init, `s` would have a norm comparable to `d`, so the first epochs would start from a
model much worse than DINOv2 (random noise in half of the similarity). Zero init gives a **safe start**: the loss
at update 0 is the DINOv2 loss, and the SD branch only grows in the directions the loss asks for. This is the
same trick as LoRA's `B = 0`, ControlNet's "zero convolution", and the zero-initialised last layer of residual
blocks (Goyal et al., 2017).

### 2.2 How the gradient flows when `out` is zero

Write `h = blocks(inp(z))` (the input of `out`) and `g = dL/ds` (the gradient arriving at `s`).

- `dL/dW_out = g h^T`. `h` is not zero (the earlier layers have random weights), so **`out` gets a gradient at
  the first update.**
- `dL/dh = W_out^T g = 0`, because `W_out = 0`. So **`inp` and `blocks` get zero gradient at the first update.**
  They start moving from the second update, once `W_out` is no longer zero.

One update of delay is harmless. This works only because `h != 0`: if both `out` and the layer before it were
zero, the gradient of each would be multiplied by the other and both would stay at zero forever.

### 2.3 The first gradient is orthogonal to DINOv2

At update 0, `f = normalize(d + s)` with `s = 0`. The Jacobian of `normalize(v)` at `v = d` (`||d|| = 1`) is
`I - d d^T`, the projection onto the space orthogonal to `d`. So

```
dL/ds = (I - d d^T) dL/df
```

The first gradient pushes `s` only in directions **orthogonal** to the DINOv2 feature of that cell. Changing `s`
along `d` would only rescale `d + s`, which the normalisation removes. In words: the SD branch starts by adding
new directions (information DINOv2 does not have), not by amplifying DINOv2. This is the theoretical reason why a
residual SD branch can improve on DINOv2 without first having to reproduce it.

---

## 3. "adapter_mix" and the initialisation `W_s = I`

`f = normalize(W_d d + W_s s + b)`, `mix` = 1x1 conv 2048 -> 1024, init `W_d = I`, `W_s = I`, `b = 0`.

### 3.1 What the mix can learn that the sum cannot

The fixed sum `d + s` adds channel k of DINOv2 to channel k of SD. The mix lets every output channel be any linear
combination of all 2048 input channels. It can:

- reweight DINOv2 channels (`W_d != I`), for example damp channels that confuse left and right;
- choose, per channel, how much SD counts (an explicit per-channel version of the gate in section 1.4);
- rotate the SD channels into the DINOv2 basis.

The cost is 2.1M extra parameters at `lr_adapter`, and `W_d` lets the model change DINOv2 features after LoRA,
which partly duplicates what LoRA already does.

### 3.2 Why `W_s = I` and not `W_s = 0`

Gradients through the mix:

```
dL/dW_s = g' s^T            (g' = gradient at the mix output)
dL/ds   = W_s^T g'
```

- **With `W_s = 0`:** `s = 0` (because `out = 0`), so `dL/dW_s = g' 0^T = 0`. And `dL/ds = 0^T g' = 0`, so `out`
  gets no gradient either. Both stay at zero forever. The point `(W_s = 0, out = 0)` is a saddle point of the loss
  where the gradient is exactly zero: plain gradient descent (and Adam) can never leave it. The whole SD branch
  would be dead weight.
- **With `W_s = I`:** `dL/ds = g'`, not zero, so `out` gets a gradient at update 1 (as in section 2.2). `W_s`
  gets zero gradient at update 1 (because `s = 0`) and starts moving from update 2, once `s != 0`.

`W_d = I` makes the initial model equal to "adapter" and therefore to DINOv2. The paired comparison
`unified_lora_large_sd` vs `unified_lora_large_sd_mix` (same seeds) then measures only what the learned merge
adds.

### 3.3 Interaction with weight decay

AdamW decoupled weight decay pulls every weight toward **zero**, not toward the initial value. For `mix` this means
`W_d` and `W_s` are slowly pulled away from `I` toward 0. With `weight_decay` 0.05, peak lr 3e-4 and about 470
updates, the shrink factor is at most `(1 - 3e-4 x 0.05)^470 ≈ 0.993`, a 0.7 % shrink: negligible, and the final
L2 normalisation removes a uniform scale anyway. A "decay toward I" (penalise `||W - I||^2`) is the theoretically
cleaner choice for a matrix that starts at the identity, but at this scale it would not change the result.

---

## 4. `sd_noise_samples` 4 (noise ensemble at evaluation only)

DIFT features are taken from the UNet after adding noise at timestep t = 261 to the latent. The feature map is a
random variable that depends on the noise draw `eps`: `z = z(x, eps)`.

- **At evaluation:** `src/models/sd.py` averages the raw up-block features over 4 fixed noise draws, then
  normalises. Averaging K independent draws divides the noise-induced variance of each feature by K (standard
  deviation by 2 for K = 4), so the feature is closer to its expectation `E_eps[z(x, eps)]`. A lower-variance
  descriptor gives more stable similarity maps. The DIFT paper uses the same ensemble.
- **At training:** 1 draw. This is cheaper (SD is the most expensive forward), and the noise acts as data
  augmentation for the adapter.

**Implication: a small train/test mismatch.** The adapter is trained on single-draw features (higher variance)
and evaluated on averaged features (lower variance, slightly smoother). Because the adapter is mostly linear and
the average is close to the expected input, the mismatch is usually benign, and it is the same situation as
dropout (noisy at training, deterministic at test). The fixed seeds keep evaluation deterministic.

---

## 5. `cross_transform` (off in the preset)

`cross_transform` is a second adapter applied after the merge. It is **cross-image**: dilated convolutions plus
bidirectional cross-attention between the source and the target feature maps, so each image's features depend on
the other image. Theoretically this can resolve ambiguity (pick the wheel that matches the source pose). It is off
in the SD presets so that the comparison with `unified_lora_large` measures the SD fusion alone (one change at a
time), and because it adds another randomly initialised trained module on top of the adapter.

---

## 6. Gradient clipping (`grad_clip`)

Code (`src/train.py`, one optimiser update):

```python
grad_norm = clip_grad_norm_([p for g in groups for p in g["params"]], cfg["grad_clip"])
optimizer.step()
```

`clip_grad_norm_` computes the **global** L2 norm over all trained parameters (LoRA, LayerNorms, adapter, mix
together). If the norm exceeds `grad_clip`, every gradient is multiplied by the same factor
`grad_clip / grad_norm`. In pair mode the gradients are first divided by the number of queries of the update, so
the norm is per query and does not depend on the batch size.

### 6.1 What clipping does with Adam

With SGD, clipping caps the step size. With Adam (AdamW here) the step is `m / (sqrt(v) + eps)`, which is
**invariant to a constant rescaling of the gradient**: if every gradient were multiplied by c forever, m and v
would scale by c and c^2, and the step would not change. So with Adam, clipping does **not** limit the step
size in steady state. What it does:

- it limits the influence of **rare spikes**: a single huge gradient would otherwise inflate `m` (the step
  direction) and `v` (which then makes the next steps too small for a long time, since `beta2 = 0.999`
  remembers about 1000 updates, longer than the whole run);
- because the norm is global, a spike in one group (for example the adapter) rescales the gradient of every group
  for that update, LoRA included.

### 6.2 Implications of moving from 10 to 15

- The current preset uses `grad_clip` 10 (from `BASE`; no preset overrides it).
- Raising it to 15 only changes updates whose norm is between 10 and 15; updates below 10 are identical, and
  updates above 15 are still clipped (less hard).
- The SD presets have about 8 times more trained parameters than the LoRA-only preset, and the adapter runs at a
  higher lr, so their global norm is naturally larger. If the logged `optim/grad_clipped` (W&B) is above about
  10-20 % of the updates, clipping is acting as a constant rescale rather than as spike protection, and a higher
  threshold such as 15 is justified. If it is near 0 %, changing the threshold has no effect.
- Decide from the logs: read `optim/grad_norm` of a finished `unified_lora_large_sd` run, and set `grad_clip`
  a little above its typical (median) value, so that only spikes are clipped.

---

## 7. EMA of the weights

### 7.1 There are two EMAs in `train.py`

1. **`ema_weights` (the real one).** `src/utils/ema.py`, `ema_decay` 0.99. It keeps a shadow copy of the trained
   weights, `shadow = 0.99 shadow + 0.01 param` after every optimiser step. Validation, `best.pt` and the final
   checkpoint use the shadow weights (`averaged_weights` swaps them in and out). This one changes the model.
2. **`ema` dict (logging only).** `0.95 * old + 0.05 * new` on the training loss statistics, for smoother W&B
   curves (`train/*_ema`). It does not touch any weight.

The one that matters for the score is the first.

### 7.2 Theory

Polyak-Ruppert averaging: with a constant (or slowly decaying) lr, SGD/Adam iterates oscillate around a minimum
because of gradient noise. The average of the iterates sits closer to the centre of the basin than any single
iterate, so it usually has lower loss and generalises better (wider, flatter point, as in SWA, Izmailov et al.,
2018). An EMA is an average that forgets old weights geometrically: weight `(1 - beta) beta^k` on the iterate k
steps ago, with an effective memory of `1 / (1 - beta) = 100` updates for `beta` = 0.99. With about 78 updates per
epoch this is about 1.3 epochs.

### 7.3 Which parameters it covers (is it "only LoRA"?)

The shadow is built from `p.requires_grad` at construction time (`EMA.__init__`). In `train.py`,
`configure_trainable` (line 296) runs before the EMA is built (line 370), so the set is exactly the trained set:

| Part | In the EMA | Correct? |
|---|---|---|
| LoRA `lora_A`, `lora_B` (blocks 18-23) | yes | yes |
| LayerNorms of blocks 18-23 and final LayerNorm (`lora_train_layernorm` True) | yes | yes, they are trained |
| Fusion adapter (`inp`, `blocks`, `out`) and `mix` | yes | yes |
| `cross_transform` (when on) | yes | yes |
| Frozen DINOv2-L weights | no | yes, they never change |
| Stable Diffusion (UNet, VAE) | no | yes, frozen and `requires_grad_(False)` |

So the EMA is **not only on LoRA**: it covers every trained parameter, and that is the correct choice. If only
LoRA were averaged, validation would combine averaged LoRA weights with live (non-averaged) adapter weights.
These two parts were trained together; mixing a 100-update-old LoRA with the current adapter gives a model that
never existed during training, and the adapter would see DINOv2 features that it was not trained with. The model
has no BatchNorm, so there are no buffers to average.

### 7.4 Correctness of the implementation

- Update after `optimizer.step()`, once per optimiser update (not per micro-batch): correct.
- Shadow initialised from the weights after any checkpoint load or `init_artifact` load: correct, the average
  starts from the weights the run actually starts from.
- Resume: `last.pt` stores both the live weights (for the optimiser) and the shadow; `best.pt` stores the shadow.
  Correct.
- `swap_in` / `swap_out` copy in place under `no_grad`, so the optimiser state still refers to the live weights
  after the swap: correct.

### 7.5 Is it the best implementation? Two theoretical limits

1. **Bias toward the initial weights early in training.** The shadow starts equal to the initial model (DINOv2 with
   `lora_B = 0` and `out = 0`). After n updates the initial weights still have weight `0.99^n` in the average:
   0.46 after epoch 0 (78 updates), 0.21 after epoch 1, 0.10 after epoch 2. So the validation of the first epochs
   measures a model that is still about half the initial one, and it underestimates the live model. By the end
   (about 470 updates, `0.99^470 ≈ 0.009`) the bias is gone, so the **final** `best.pt` is not affected much;
   the early-epoch val PCK and early stopping are. The standard fix (TensorFlow `ExponentialMovingAverage`,
   timm with warmup) is a decay warm-up:
   `decay_n = min(0.99, (1 + n) / (10 + n))`, so the average starts short and grows to 100 updates.
2. **EMA of LoRA factors is not EMA of the LoRA update.** The weight change of a LoRA layer is the product
   `Delta W = (alpha / r) B A`. The EMA averages `A` and `B` separately, and
   `EMA(B) EMA(A) != EMA(B A)` in general: the difference is the covariance of the B and A trajectories over the
   averaging window. Near convergence, when the iterates move little, the difference is small, and averaging the
   factors is what timm/PEFT-style training does in practice. Averaging the merged `Delta W` would be exact but
   needs a full 1024 x 1024 (or 1024 x 4096) matrix per wrapped layer, about 100 times the LoRA memory. The same
   remark applies to the adapter (GELU makes it non-linear in its weights): weight averaging relies on the
   iterates staying in one basin, which holds at this lr.

Verdict: the implementation is correct and covers the right parameters. The one change with a clear theoretical
benefit is the decay warm-up of point 1. It matters for the per-epoch validation and early stopping, much less for
the final checkpoint.

---

## 8. Can the bottleneck blocks after SD be simplified?

Short answer: yes. The theory says the two blocks are probably more than the SD branch needs. The cheapest
simplification is to reduce the depth (`adapter_layers` 2 -> 1 or 0). It needs no code change. Simplifying the
inside of a block saves little, because most adapter parameters are in `inp`, not in the blocks. Only a paired
run can decide, because the gain of SD itself is not measured yet (D-009).

### 8.1 What each part of the adapter really contributes

| Part | Parameters | What it adds | Is it needed? |
|---|---:|---|---|
| `inp` 1x1 3200 -> 1024 | 3,277,824 (50 %) | channel selection and dimension match | yes, `d + s` needs 1024 channels |
| 2 bottleneck blocks | 2,231,296 (34 %) | non-linearity (GELU) + 5 x 5-cell spatial context | partly, see below |
| `out` 1x1 1024 -> 1024, zero init | 1,049,600 (16 %) | safe start (`s = 0` at update 0) | yes, for the zero init |

**The spatial context of the 3x3 is mostly redundant.** SD features come from the UNet decoder, whose receptive
field already covers most of the image (many down/up-sampling stages plus self-attention). Resizing to the
46-cell grid adds more mixing: up-block 0 (23 cells) is upsampled, so neighbouring cells already share
information, and up-block 2 (92 cells) is downsampled, which averages 2 x 2 cells. A 3x3 convolution on top adds
little context that SD does not already have, and it can widen the similarity peak (section 1.3).

**The non-linearity is the real job of the blocks.** Without any block, `out(inp(z))` is a product of two linear
maps, so the adapter is a single linear projection of SD to 1024 channels. That is the same family as the PCA
projection of SD+DINO (Zhang et al., 2023), only learned. The GELUs make the adapter able to gate or combine SD
channels non-linearly. One block already gives this, and two blocks give a deeper composition of the same kind.

**Data size.** The run has about 470 optimiser updates. With so few updates, a larger adapter mostly adds
overfitting risk and noise between seeds. The residual design and the zero-initialised `out` make extra blocks
harmless at the start, but they do not limit what the blocks can memorise later.

### 8.2 Option A: fewer blocks (config only, recommended first)

`adapter_layers` is a key of `BASE` (`src/config.py`). With 0 blocks, `nn.Sequential()` is empty and acts as the
identity, so the code already supports all three settings:

| `adapter_layers` | Adapter parameters | Receptive field | Non-linear? |
|---:|---:|---|---|
| 2 (current) | 6,558,720 | 5 x 5 cells | yes, 2 stacked blocks |
| 1 | 5,443,072 | 3 x 3 cells | yes, 1 block |
| 0 | 4,327,424 | 1 cell | no: one learned linear projection |

- **1 block:** keeps everything the blocks offer in theory (non-linearity, some context) with half the block
  parameters and a narrower receptive field. It is the most likely "same score, simpler model".
- **0 blocks:** the linear baseline. If it matches 2 blocks, the non-linearity is not used and the adapter is only
  a learned projection. Note: `out` and `inp` are then two linear maps in a row (a factorised 3200 -> 1024 map).
  This is redundant in capacity, but `out` is still needed for the zero init.

### 8.3 Option B: a simpler inside of each block (needs a code change in `src/models/fusion.py`)

| Variant of one block | Parameters per block | Change |
|---|---:|---|
| current: 1x1 down, 3x3, 1x1 up | 1,115,648 | none |
| no 3x3: 1x1 down, GELU, 1x1 up (per-cell MLP) | 525,568 | removes spatial context |
| depthwise 3x3 (`groups=256`) in the middle | 528,128 | keeps context, per channel only |
| narrower: `adapter_dim` 128 (config only) | 410,880 | smaller bottleneck |

- **Per-cell MLP** matches the argument in 8.1: keep the non-linearity, drop the context SD already has. It also
  removes the border-padding effect and any blurring of the peak.
- **Depthwise 3x3** keeps a small spatial smoothing for about 0.5 % of the cost of the full 3x3.
- **`adapter_dim` 128** is a config change, but it narrows the non-linear part that matters most (8.1). Try it
  only after Option A.

These variants save at most about 1.2M of the 6.56M parameters, because `inp` (3.28M) stays. If the goal is a
much smaller adapter, the effective change is to make `inp` low-rank: 3200 -> 256, then the blocks at width 256,
then `out` 256 -> 1024. `inp` becomes 819,456 and `out` 263,168. This is a code change, and it also lowers the
rank of the SD contribution to 256, which is a stronger constraint against overfitting.

### 8.4 Recommended order

1. `unified_lora_large_sd` (2 blocks) is the baseline, seeds 0 and 1.
2. The same preset with `adapter_layers` 1, same seeds. Compare with `scripts/compare_runs.py`.
3. If 1 block is within noise of 2 blocks (gain inside the 95 % interval), keep 1 block, the simpler model. Then
   try `adapter_layers` 0. If 0 blocks is also within noise, the SD branch only needs a linear projection.
4. Change the inside of the blocks (Option B) or `inp` only if the depth runs show that the adapter overfits:
   train loss keeps falling while val PCK stays flat or drops.
