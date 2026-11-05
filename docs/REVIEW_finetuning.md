# Review: first DINOv2 fine-tuning run

Run reviewed: `20261104-155818_train` (W&B project `sapienza-aml-semantic-correspondence`), Colab T4,
code of commit `4615170` (branch `v2`). Numbers below come from the run's W&B history (788 logged updates).

The review has two parts:

1. **How the fine-tuning works**: each question from the run notes, explained with the code and the numbers of this run.
2. **Takeaways from the run and W&B**: what the run showed, and the changes that follow (charts to delete or add,
   training settings). Part 2 is applied in `src/config.py` and the code (2026-11-05). The optimiser settings in
   2.4 (learning rates, warmup, clip) were then replaced by AdamW + OneCycleLR, lr_head 1e-3, clip 5: current
   values in [RUN.md](RUN.md) section 8.

Related: [PRD_finetuning.md](PRD_finetuning.md) (design and hyperparameter choices), [RUN.md](RUN.md) (how to run).

## The run in one table

| | frozen DINOv2 (no training) | epoch 0 | 1 | 2 | 3 | 4 | **5 (best.pt)** |
|---|---|---|---|---|---|---|---|
| val PCK@0.10 | 40.4 | 47.2 | 57.6 | 58.1 | 62.0 | 62.3 | **62.9 ± 4.5** |
| val PCK single target | 35.1 | 43.6 | 52.6 | 54.5 | 58.3 | 58.3 | 58.8 |
| val PCK multi target | 45.0 | 50.4 | 62.0 | 61.2 | 65.3 | 65.7 | 66.5 |
| val PCK small / medium / large | 61.8 / 38.0 / 34.6 | 56.6 / 46.4 / 44.6 | 60.5 / 56.6 / 57.4 | 55.3 / 59.0 / 58.3 | 54.0 / 65.1 / 62.6 | 54.0 / 66.3 / 62.1 | 54.0 / 66.3 / 63.5 |
| val loss | 8.87 | 5.42 | 5.04 | 4.91 | 4.85 | 4.81 | 4.80 |
| train loss (mean of the epoch) | | 6.81 | 4.82 | 4.52 | 4.32 | 4.24 | 4.18 |
| train softmax entropy | | 4.10 | 4.78 | 4.52 | 4.32 | 4.25 | 4.19 |
| grad norm (median) | | 26.8 | 11.1 | 11.0 | 11.0 | 10.6 | 10.8 |

Setup: ViT-B/14 at 644 px, blocks 10 and 11 + final LayerNorm + linear head trained (14.77M parameters),
micro-batch 16 × 1 accumulation step = effective batch 16, 2,042 training queries → 127 updates per epoch,
889 updates planned over 7 epochs. Training ended about 25 updates into epoch 6, so epoch 6 was never validated
and `best.pt` is epoch 5. About 10 minutes per epoch including validation. An `evaluate --run` of `best.pt` was
logged afterwards in the same W&B run (step 788, job type `evaluate`), with the matching sweep of 2.1.

**+22.5 PCK points over the frozen model.** Almost all of the gain is in the first 4 epochs. **A further +3.1 is
available without any training:** `best.pt` scored with `soft_window = 11`, `soft_temperature = 0.02` reaches
66.0 (2.1, item 7).

---

# Part 1: How the fine-tuning works

## 1.1 What blocks 10 and 11 are

DINOv2 ViT-B/14 turns an image into features in three stages:

1. **Patch embedding.** The 644 × 644 image is cut into 14 × 14 pixel patches: 644 / 14 = 46, so 46 × 46 = 2,116
   patches. Each becomes a 768-number vector (a *token*). A CLS token is added for the whole image.
2. **12 transformer blocks, numbered 0 to 11.** All have the same structure; each one rewrites every token using
   the others:
   - LayerNorm → **self-attention** with 12 heads (each token looks at all 2,117 tokens and mixes in what is
     relevant) → LayerScale → added back to the input (residual);
   - LayerNorm → **MLP** 768 → 3,072 → 768 with GELU → LayerScale → residual.
3. **Final LayerNorm.** Its output for the 2,116 patch tokens is the 46 × 46 × 768 feature map we match.

One block has 7,089,408 parameters:

| part | parameters |
|---|---|
| attention q, k, v (3 × 768 × 768 + biases) | 1,771,776 |
| attention output projection | 590,592 |
| MLP 768 → 3,072 | 2,362,368 |
| MLP 3,072 → 768 | 2,360,064 |
| 2 LayerNorms + 2 LayerScales | 4,608 |

**Blocks 10 and 11 are the last two.** Trained here: 2 × 7,089,408 + head 590,592 + final LayerNorm 1,536 =
14,770,944, the `trainable_params_M = 14.77` shown in W&B. Blocks 0–9 and the patch embedding (≈ 72M parameters)
stay frozen.

Why the last ones:

- **They make the descriptor we match.** Matching reads the output of block 11 (`feature_layer = -1`), so the top
  blocks change the descriptor directly.
- **The lower blocks are already good, and are what makes DINOv2 general.** Early blocks encode edges, texture and
  position; the top blocks encode object parts. With only 2,042 training queries, changing the lower blocks would
  mostly fit these 392 pairs and lose generality (overfitting, forgetting).
- **Memory and speed.** Gradients only flow back to block 10. The activations of blocks 0–9 are not stored, so a
  micro-batch of 16 fits in 5.8 GB of the T4's 15 GB.
- **Evidence.** 2 blocks was the best setting for ViT-B on SPair-71k (PRD §6.5).

Each trained part has its own learning rate (layer-wise decay 0.75): head 1e-4, final LayerNorm 1e-5, block 11 1e-5,
block 10 7.5e-6. The deeper a block, the more general it is, so the smaller its learning rate.

## 1.2 Are any weights initialised randomly? No

- **Backbone:** the pretrained DINOv2 weights, at a pinned Hub commit (`model_revision`).
- **Head** (`head = "linear"`, `src/models/head.py`): a 1×1 convolution 768 → 768 initialised as the
  **identity** (weight = identity matrix, bias = 0). Its output is L2-normalised. So at update 0 the model gives
  exactly the frozen DINOv2 features, and training starts from the 40.4 baseline instead of from noise.
- The `mlp` head (not used) has a randomly initialised first layer, but its last layer starts at zero, so its
  output also starts as the identity.

Randomness exists only in the data: shuffle order (`seed + epoch`), augmentation, label jitter, and the random
images of the batch-size probe (the code re-seeds after the probe, so the probe does not change the run).

## 1.3 Micro-batch vs effective batch

- **Micro-batch:** the samples in one forward + backward pass on the GPU. Limited by GPU memory.
  `batch_size = "auto"` finds the largest that fits (`src/utils/batch_size.py`).
- **Effective batch:** the samples behind one optimiser update = micro-batch × accumulation steps. This is what the
  optimisation "sees". With accumulation, the losses of several micro-batches are added (each divided by the
  number of steps) before one update.

In this run the micro-batch of 16 fitted, so there was 1 accumulation step: micro = effective = 16. The probe
stops at `effective_batch`; it does not try 32, so 16 means "16 fits", not "16 is the maximum".

Accumulating gives the same gradient as one large batch, because ViTs use LayerNorm (computed per token) and no
BatchNorm (computed across the batch). Only time differs. A smaller micro-batch with more accumulation steps gives
the same training, only slower. Why 16, and not more: 2,042 / 16 = 127 updates per epoch. A larger batch means
fewer updates for the same data (PRD §5.3).

## 1.4 The loss, and what softmax entropy measures

Per query (`src/models/loss.py`):

1. Take the source feature at the query point, `f_s` (768 numbers).
2. Compare it with each of the 2,116 target cells: logit `z_j = cos(f_s, F_t[j]) / T`, with `T = 0.03`.
3. Softmax over the cells: `p_j`, the predicted probability that cell j is the match.
4. Target distribution `t`: a Gaussian bump (σ = 1 cell) on each of the K accepted targets, each bump weighing 1/K.
5. Loss = cross-entropy `−Σ_j t_j · log p_j`.

Cross-entropy splits into two terms:

    loss = H(t) + KL(t ‖ p)

`H(t)` is the entropy of the target itself. It does not depend on the model, so **the loss can never go below
H(t)**: a loss of 0 is impossible. Only the KL part can be trained away.

`train/softmax_entropy` = `H(p)`, the entropy of the prediction. It measures how spread out the prediction is
(not trained, only logged). It is low when all the mass is on a few cells (sharp, confident) and high when the
mass is spread out. Uniform over 2,116 cells would be ln 2,116 = 7.66.

**The floor, computed on the 46 × 46 grid:**

| target | H(t) (nats) |
|---|---|
| 1 accepted target, σ = 1 cell | 2.84 |
| 1 target with label jitter 0.5 cell (σ ≈ 1.12) | 3.06 |
| 2 targets | 3.53 (= 2.84 + ln 2) |
| 3 targets | 3.94 |
| 4 targets | 4.22 |

The training queries: 47.5% have 1 target, 39% have 2, the rest 3–6 (mean ln K = 0.45). So the average floor is
≈ 3.3 nats, ≈ 3.5 with jitter. **The best possible model has H(p) ≈ H(t) ≈ 3.3–3.5.** Below that it is
overconfident; above, too spread out. This run ends at 4.19, so there is still about 0.7 nats of real error.

### Reading the entropy as a number of cells

The logger uses natural logarithms (`log_softmax`), so **e^H is the effective number of equally likely cells**: a
distribution with entropy H is as spread out as a uniform choice among e^H cells.

| state | H (nats) | e^H (cells) |
|---|---|---|
| ideal, 1 target (Gaussian σ = 1 cell; 2πe·σ² ≈ 17) | 2.84 | 17 |
| ideal, 1 target with label jitter | 3.06 | 21 |
| ideal, average over the training queries | 3.3–3.5 | 27–33 |
| this run, mean of the last 50 updates | 4.20 | 66 |
| this run, peak of the 50-update moving average (update 128) | 5.25 | 191 |
| this run, peak of a single update (update 92) | 5.83 | 340 |
| uniform over all 46 × 46 cells | 7.66 | 2,116 |

At the end the prediction is about **2–2.5× wider than the ideal target**. At the moving-average peak it was
about 6× wider.

**66 effective cells does not mean 66 separate plausible matches.** Entropy only measures how spread the mass is,
not its shape. The same 4.2 nats can be:

- **one smooth peak:** 66 cells is a disc of radius ≈ 4.6 cells, ≈ 64 px of the 644-px image, around one location.
  The match is right but not precise. That fits near misses being the largest error left (1.12).
- **several separate peaks:** for example the left and right eye, or all four wheels. For a multi-target query
  that is correct (the floor already allows K peaks: 2 targets = 2 peaks of 17 cells). For a single-target
  query, it is the start of a wrong keypoint.
- a mix of both.

Entropy alone cannot tell these apart. A measure that can: **the probability mass inside a small window around
the most likely cell** (for example 5 × 5 cells). It is close to 1 for one peak, however wide its flanks, and
falls when mass sits on other peaks. Proposed as an optional chart in 2.2.

### Two softmaxes: training vs inference

The model uses a softmax in two places, with different settings:

| | training loss (`src/models/loss.py`) | inference soft-argmax (`src/models/matching.py::locate`) |
|---|---|---|
| over | all 2,116 cells of the 46 × 46 map | a window of `soft_window` cells around the best cell, on the 2× upsampled map (92 × 92) |
| temperature | `loss_temperature` = 0.03 | `soft_temperature` = 0.04 (BASE) |
| output | a probability per cell, compared with the target | the weighted mean position inside the window: the prediction |

What the training teaches (a peak about 2× wider than ideal) changes which inference window works best:

- `soft_window = 5` upsampled cells = 2.5 cells of the feature grid. On a peak about 9 cells across, it sees only
  the top, so the averaged position follows small noisy bumps.
- `soft_window = 11` = 5.5 grid cells covers most of the peak, so the weighted mean lands nearer its centre.
- `soft_temperature = 0.02` sharpens the weights inside the window, so the flanks pull the estimate less.

The sweeps agree:

- on the frozen model (sharper, entropy 2.4–3.3), windows 7–9 were best;
- on the fine-tuned model (broader), window 11 with T = 0.02 is best: **+3.1 PCK (62.9 → 66.0)**, and small
  objects go back from 54.0 to 61.8 (2.1, item 7).

This is a plausible link, consistent with both sweeps, not a proof.

## 1.5 "Multiple positives raise the entropy floor"

The note says: with K positives, the best distribution spreads over all K, so the minimum entropy is ≈ log K. If
the sampler gives more positives over time, entropy rises without anything being wrong. This applies here only in
part:

- **What holds:** a query with K accepted targets has a floor of H(single) + ln K (table above). Multi-target
  queries always have a higher entropy and a higher loss.
- **What does not hold:** the note is written for a *batch-wise contrastive* loss (InfoNCE), where the positives
  are other samples of the batch. Here each query has its own softmax over the cells of its own target image;
  other samples in the batch play no part. The positives are the K accepted targets fixed in `train.csv`, and the
  sampler is a uniform shuffle, so the floor does not drift during training. Augmentation can drop a target
  that leaves the image, which lowers K slightly; it never raises it.

So the entropy rise in this run (1.8) is **not** a floor effect. It is the model flattening its prediction.

## 1.6 Loss by group: small, single, multi

The groups have different floors and are measured differently, so **a loss value does not compare across groups**:

- **Single vs multi.** Multi-target queries have a floor about +0.7 nats higher (ln 2 for two targets), and their
  loss asks for the mass to be split over *all* targets. PCK only needs the nearest one. Hence in this run
  **val loss multi 5.10 > single 4.46, while val PCK multi 66.5 > single 58.8**: multi is the better group despite
  the higher loss. Within a group, the loss going down over time is what counts.
- **Size groups.** The loss is counted in grid cells (the same 46 × 46 grid whatever the object size); PCK is
  counted in fractions of the object size L.
  - Small objects (L < 100 px) cover few cells. A bump of σ = 1 cell covers much of the object, so the loss is low
    (3.73). But the PCK tolerance (< 0.1 L, under 10 px) is smaller than one cell, so PCK is the lowest (54.0).
  - Large objects: a wrong part is many cells away, so the loss is high (5.12), but the tolerance is generous.
  - The size-group loss therefore says little about the score. Removing it from W&B and keeping PCK by size
    (part 2) is right.

## 1.7 The loss EMA

`train/loss_ema` (`src/train.py`): `ema ← 0.95 · ema + 0.05 · loss of this update`, starting at the first update's
loss.

- **Memory:** about the last 1 / (1 − 0.95) = 20 updates (half-life ≈ 13.5 updates), about 1/6 of an epoch.
- **Why it is needed:** with 16 samples per update, the raw loss is noisy. Its spread in epoch 5 (std 0.17) is
  larger than the whole epoch-4 → epoch-5 gain (0.06). The raw curve cannot show the trend; the EMA can.
- **Lag:** a change shows fully only 20–40 updates later. When the loss falls fast (warmup), the EMA sits above the
  real loss. It is not reset at epoch boundaries.
- **What it is not:**
  - It is the training loss under augmentation and label jitter, so it does not compare with `val/loss` (no
    augmentation).
  - No decision uses it: early stopping and `best.pt` use val PCK.
  - It is stored in the run data, so it is the same in every W&B view, unlike the W&B smoothing slider.
- **A train softmax entropy EMA:** yes, it is possible: the same formula on `train/softmax_entropy` (its noise is
  similar, std 0.16 in epoch 5). Proposed as `train/softmax_entropy_ema` in part 2.

## 1.8 Warmup, and why the softmax entropy rises

**Schedule.** Warmup = max(30, 10% × 889) = **89 updates** (0.7 epochs). The learning rate rises linearly from
1/89 of the peak to the peak (head 1.1e-6 → 1e-4, block 11 1.1e-7 → 1e-5), then follows a cosine down to 1% of the
peak at update 889.

**What happened:**

| updates | softmax entropy | train loss | grad norm |
|---|---|---|---|
| 1–40 | 2.4–3.3 | 8.3–9.1 | 36–44 |
| 40–92 | rises to **5.83 at update 92** | 8.3 → 5.2 | 38 → 9 |
| end of epoch 1 → end of run | 4.78 → 4.19 | 4.8 → 4.2 | ≈ 11 |

The entropy peak (update 92) falls exactly at the end of the warmup (update 89), when the learning rate is highest.
Single updates are noisy (std ≈ 0.16), so the trend reads better on a **50-update moving average**: it peaks at
**5.25 around update 128**, then falls to **4.20** over the last 50 updates.

**One plausible reading, in three phases.** The curves are consistent with it; entropy alone does not prove it.

1. **At the start, predictions are sharp but often wrong.** Low entropy is not good in itself: it means
   confidence, not accuracy.
2. **Early adaptation: the model removes some of that wrong confidence.** Entropy rises while the loss improves
   (8.3 → 5.2).
3. **Later, it learns more discriminative matches and sharpens again.** Entropy 5.25 → 4.20 while val PCK goes
   47 → 63.

**Why.**

1. **The frozen model is confidently wrong.** At T = 0.03 the frozen DINOv2 features give a sharp prediction
   (entropy 2.4–3.3, below the 3.3–3.5 floor) that is often on the wrong cell. Cross-entropy punishes that very
   hard (log p of the right cell is very negative): loss ≈ 9.
2. **The fastest way to lower such a loss is to flatten p.** Spreading the mass also puts some on the right cells.
   So the entropy has to rise at least from ≈ 2.7 to the floor (≈ 3.4). That part of the rise is correct
   behaviour.
3. **The overshoot.** The entropy went to 5.8, far above the 4.2 the model ends at, so it flattened more than
   needed. It then took about 5 epochs to sharpen again. The overshoot coincides with the learning-rate peak.
   The likely main driver is the head: at 1e-4 it is 10× the backbone rate, and since it is one 768 × 768 matrix
   applied to *every* cell, a change to it moves all features at once and can flatten the whole similarity map
   in a few updates.

So the reading in the notes is right: **a slower warmup with a lower peak learning rate should reduce the entropy
overshoot.** Changes in part 2: warmup 20%, and both learning rates halved (head 1e-4 → 5e-5, backbone
1e-5 → 5e-6). One option not in the change list:
*LP-FT* (Kumar et al. 2022, "Fine-Tuning can Distort Pretrained Features"): train the head alone first, then
unfreeze the blocks.

## 1.9 The gradient norm: why ≈ 10, and what clipping at 1 does

**What is logged.** `optim/grad_norm` is the value returned by `clip_grad_norm_`: the total L2 norm of the gradient
over all 14.77M trained parameters, **before** clipping and after removing the AMP scale.
`optim/grad_clipped = 1` when it is above `grad_clip`.

**In this run:**

- epoch 0: median 26.8 (36–44 in the first 60 updates);
- after the warmup: median 11.0, 95th percentile 14.5, 99th 17.3, maximum 19.6;
- **clipped in 99–100% of the updates of every epoch.**

**Why it is so large.**

1. **The temperature.** The logits are cos / T with T = 0.03, so every gradient through them is multiplied by
   1/T = 33. With T = 1 the same model would show a norm of ≈ 0.3. The size of the norm comes from how the loss
   is written; it is not a sign of instability.
2. **Why it stays at ≈ 10 instead of going to 0.** The gradient on the logits is (p − t) / T, and p never equals t:
   - label jitter moves t at every update;
   - augmentation changes the images;
   - 2,042 queries cannot all be fitted.

   This leftover error is noise. Averaging over 16 samples only divides it by about √16 = 4. The plateau at ≈ 10 is
   this noise floor.
3. **Why it falls from 40 to 10 during warmup.** ‖p − t‖ is largest when p is a sharp peak on the wrong cell (the
   frozen start). As p stops being confidently wrong (the entropy rise of 1.8), the gradient shrinks.

**Three infinite values** (updates 2, 5 and 18) are fp16 overflows at the GradScaler's starting scale. The
scaler skipped those updates and lowered its scale from 32,768 to 4,096 (`optim/amp_scale`). This is normal AMP
behaviour, not a NaN problem (`train/nonfinite_batches` stayed 0).

**What `grad_clip = 1` does with AdamW.** Clipping multiplies the whole gradient by 1/‖g‖ (≈ 1/11 here). AdamW then
divides each gradient by its running RMS (√v), so a constant rescaling cancels out. **The step size is not 11×
smaller: the effective learning rate is about the same.** What is lost:

- **The safety guard.** When every update is clipped, a real spike (one bad batch) is treated like any normal
  update. Clipping protects against nothing.
- **The weighting between batches.** Every update becomes unit-length, so a batch with large errors pushes no
  harder than an easy one (normalised-gradient training).
- **The metric.** `grad_clipped` is always 1 and tells nothing.

The likely effect on PCK is small, but one run cannot prove it either way.

**Fix.** Use clipping as a spike guard above the normal range: **`grad_clip = 20`**, just above the largest norm
after the warmup (19.6). A clip at 10 would still cut more than half of the updates (median 11).

The clipped updates in epoch 0 are the early, confidently-wrong phase, where extra braking helps the warmup.
Another option is an adaptive clip (for example 2 × the running median of the norm); the fixed value is simpler
and enough here.

## 1.10 Safety guards for fine-tuning DINOv2

Already in the code:

- small backbone learning rate with layer-wise decay;
- only 2 blocks trained;
- identity-initialised head (start = baseline);
- warmup;
- GradScaler skips overflowed updates; a non-finite loss skips the batch;
- `best.pt` only on a better val PCK; early stopping;
- no weight decay on LayerNorm weights and biases.

The weight decay of 0.05 pulls the pretrained weights towards 0, but at lr 1e-5 over 889 updates that shrinks
them by only ≈ 0.04%, so it is harmless.

Proposed (part 2):

1. `grad_clip = 20` as a real spike guard (1.9).
2. **Embedding variance charts** to see feature collapse (1.11).
3. A gentler warmup (1.8).

Options for later, not in the change list:

- **L2-SP** (Li et al. 2018): a penalty on the distance of the trained blocks from the pretrained weights, so the
  features stay close to DINOv2.
- **LP-FT** (1.8).

## 1.11 Embedding variance across the batch (new chart)

**Feature collapse** = the features become alike. The cosine similarities then become equal, the softmax flattens
(entropy rises) and PCK drops. A collapse would look exactly like the entropy rise of 1.8, so a chart is needed to
tell the two apart.

For unit-length vectors, the total variance is `1 − ‖mean‖²`: 0 when all vectors are identical, close to 1 when they
point in all directions. Two charts:

- `train/embed_var_batch`: over the 16 source query descriptors of an update. Do different queries still get
  different descriptors?
- `train/embed_var_cells`: over the 2,116 cells of each target feature map, averaged over the batch. Is the
  target map still contrasted, or flattening?

Both are computed from features already in memory (detached, no extra cost worth counting), every update.

How to read them: a level close to its update-0 value (= the frozen DINOv2), or a slow drift, is healthy. A
steady fall together with a rising entropy is collapse: lower the head learning rate or stop the run.

## 1.12 The error-type bar plot: how it is built, and the causes

Each validation query gets one type, checked in this order (`src/metrics.py::_error_types`). Here d = distance from
the prediction to the nearest accepted target ÷ L, the object size:

| type | rule | typical case |
|---|---|---|
| correct | d < 0.10 | |
| wrong_keypoint | not correct, and the prediction is within 0.10 L of an accepted target of **another query of the same pair** | left eye ↔ right eye, front ↔ back wheel, another labelled part |
| near_miss | 0.10 ≤ d < 0.20 | right part, not precise enough |
| other_miss | 0.20 ≤ d ≤ 0.50 | on the object, wrong (unlabelled) part |
| far_miss | d > 0.50 | background or another object |

The bar plot counts the 453 validation queries per type at each validation.

| type (% of queries) | frozen | fine-tuned epoch 5 | change |
|---|---|---|---|
| correct | 40.4 | 62.9 | +22.5 |
| near_miss | 20.1 | 16.1 | −4.0 |
| wrong_keypoint | 9.1 | 9.9 | +0.8 |
| other_miss | 17.7 | 9.1 | −8.6 |
| far_miss | 12.8 | 2.0 | −10.8 |

**What the bars show:**

- **far_miss and other_miss** are where fine-tuning helped most (−19 points together): the model no longer
  matches the background or the wrong region.
- **near_miss is now the largest error (16%).** It is about precision:
  - The grid cell is 14 px of the resized image.
  - The label bump (σ = 1 cell) plus the jitter (0.5 cell) teach a blur of about one cell.
  - Small objects have a tolerance below one cell, so they suffer most. This matches small objects going *down*
    from 60.5 (epoch 1) to 54.0 while medium and large went up. The frozen model scored 61.8 on them. n = 76,
    ± 11, so the drop is not certain, but the direction is consistent.
  - Remedies:
    - first, the inference window: `soft_window = 11`, `soft_temperature = 0.02`, with no training at all, already
      takes `best.pt` from 62.9 to 66.0 and small objects from 54.0 to 61.8 (1.4, 2.1);
    - then higher resolution (the 756 / 840 sweep), lower label jitter, then label σ 0.75.
- **wrong_keypoint did not move (9–10%).** These are mostly symmetric parts (left/right) and repeated parts. The
  pair-flip augmentation keeps left and right consistent but does not teach the difference; DINOv2 features
  are known to be weak there. Single-target queries have more of them, because there the "other" part is never
  accepted. Remedy measured: `flip_test`. It gave +3.1 on the frozen model, but on the fine-tuned `best.pt` only
  +0.7 at best (66.7 vs 66.0 with window 11), well inside the interval. Fine-tuning with the pair flip already
  took most of that gain.

## 1.13 `val/pck@0.1_ci95`

The half-width of the 95% interval of val PCK: `1.96 × √(p(1 − p) / n) × 100`, with n = 453 queries. For
p = 62.9%: ± 4.45 points. Meaning: another validation set of the same size, from the same data, would give a score
within ± 4.5 of this one about 95% of the time.

Two caveats:

- **It is optimistic.** Queries of the same image pair are correlated (453 queries come from only 78 pairs), so the
  true interval is wider.
- **It is pessimistic for comparisons on the same queries.** Two settings scored on the same 453 queries share
  most of their noise (a *paired* comparison), so a difference smaller than ± 4.5 is not automatically noise. It
  just cannot be claimed from this number alone.

Here: epochs 3 → 5 gain +0.9, well inside the interval. The curve has reached its plateau.

## 1.14 Is the augmentation heavy?

Per image (source and target are augmented separately, `src/augment.py`):

| augmentation | probability | strength |
|---|---|---|
| pair flip (both images) | 0.5 | |
| random crop | 0.5 | side 70–100% |
| rotation | 0.5 | ± 15° |
| colour (brightness, contrast, saturation) | 0.8 | ± 30%, hue ± 0.05 |
| grayscale | 0.1 | |
| blur | 0.1 | σ 0.1–2 |

How often an image is left as it is:

- geometry unchanged: 0.5 × 0.5 = 25%;
- colour unchanged: 0.2 × 0.9 × 0.9 = 16%;
- **completely unchanged: 4%.**

Since the two images are augmented independently, the model almost never sees an original pair. The reference
pipeline (Lang) uses only the pair flip.

What the run says:

- The augmented training loss (4.18) is *below* the validation loss (4.80, no augmentation). The model still fits
  the training data well, so the augmentation does not block learning.
- It does not prevent the plateau after epoch 3 either.
- The crop (down to 70%) and the rotation (up to 15°) move parts by several cells, which works against the
  precision that near misses and small objects need.

**Verdict:** medium-heavy for 2,042 queries and 5 epochs. Making it less invasive is reasonable (values in 2.4).

## 1.15 Are 5 epochs enough?

- Val PCK by epoch: 47.2, 57.6, 58.1, 62.0, 62.3, 62.9. After epoch 3 the gains (+0.2, +0.7) are inside the
  interval.
- With a cosine schedule, the late low-learning-rate phase is what polishes the model. With `epochs = 5` the cosine
  ends at epoch 5, so that phase moves earlier instead of being lost.
- Augmentation means each epoch shows new versions of the images, so more epochs are not pure repetition. But the
  curve shows the plateau anyway.

**Yes: 5 epochs** (635 updates, ≈ 50 minutes instead of ≈ 70). Early-stop patience goes to 2, since 3 could almost
never trigger in 5 epochs. One check: part 2 halves both learning rates and lengthens the warmup, which slows
learning. So if val PCK is still rising clearly at epoch 4, go back to 6–7 epochs.

---

# Part 2: Takeaways from the run and W&B

## 2.1 What the run showed

1. **Fine-tuning works:** 40.4 → 62.9 val PCK@0.10 (+22.5, interval ± 4.5). Gains come mostly from far and other
   misses (1.12).
2. **Plateau after epoch 3:** 5 epochs are enough (1.15).
3. **The softmax entropy rises during the warmup** (2.6 → 5.8, peak at the learning-rate peak), then comes back to
   4.2. Part of the rise is needed (the frozen model is overconfident); the overshoot calls for a slower warmup and
   a lower head learning rate (1.8).
4. **The gradient norm settles at ≈ 11, and `grad_clip = 1` clips every update.** The norm is large because of
   T = 0.03. Clipping at 1 does not slow AdamW much, but it removes the spike guard (1.9). New value: 20.
5. **Small objects get worse** (61.8 frozen → 54.0) while medium and large improve by about 28 points. Likely
   cause: the label blur of ≈ 1 cell, plus crop and rotation (1.12, 1.14).
6. **The error-type bar plot is the useful view of the misses** (1.12); the separate `*_pct` curves repeat it.
7. **The inference settings are now the cheapest gain.** The matching sweep on `best.pt` (W&B `val/sweep`, step
   788):

   | `flip_test` | `soft_window` | `soft_temperature` | val PCK@0.10 | small objects | |
   |---|---|---|---|---|---|
   | False | 5 | 0.04 | 62.9 | 54.0 | current BASE |
   | False | 11 | 0.02 | **66.0** | **61.8** | **recommended** |
   | False | 11 | 0.04 | 65.8 | 65.8 | |
   | True | 11 | 0.04 | 66.7 | 65.8 | best overall |

   **+3.1 points from two inference settings, against +0.9 from the last two training epochs together**
   (62.0 → 62.9). `flip_test` would add only 0.7 more, well inside the interval, and costs an extra source
   forward pass per query, so it stays off.

   Caveat: the sweep picks the best of 20 settings on the same 453 queries that chose the epoch, so the 66.0 is a
   little optimistic. The gain has the same sign for every window ≥ 9, which makes it unlikely to be luck.

## 2.2 W&B charts: what to delete, keep and add

**Delete:**

| key | reason |
|---|---|
| all W&B **system** charts (GPU use, memory, temperature, power, CPU, disk, network) and the console-log capture | not of interest. `wandb.init(settings=wandb.Settings(x_disable_stats=True, console="off"))` (checked on wandb 0.31) |
| `system/samples_per_s`, `system/peak_gpu_gb` | GPU state, not of interest. The probe's peak stays in the run summary (`probe_peak_gb`) |
| `train/epoch` | only checks that the updates per epoch add up |
| `val/n_queries`, `val/n_pairs` | constant (453, 78); they belong in the summary |
| `val/best_pck` | the run summary already holds `best_pck` |
| `val/near_miss_pct`, `val/far_miss_pct`, `val/other_miss_pct`, `val/wrong_keypoint_pct`, `val/correct_pct` | the error-type bar plot shows them; `correct_pct` is the same number as `val/pck@0.1` |
| `train/loss/small`, `train/loss/medium`, `train/loss/large` and the same three `val/loss/...` | not comparable with PCK (1.6). Keep `loss/single` and `loss/multi`, for train and val |

**Also proposed for deletion (not in the notes; confirm):**

| key | reason |
|---|---|
| `val/score` | = `val/pck@0.1` / 100 |
| `val/epoch`, `val/sec_per_query`, `val/flip_rate` | bookkeeping; `flip_rate` is always 0 while `flip_test` is off |
| `batch/micro`, `batch/grad_accum_steps`, `batch/effective`, `batch/samples_seen` | flat lines without an `effective_batch_schedule`; to the summary instead, logged only when a schedule is set |
| `train/nonfinite_batches` | always 0; the console warning stays |

**Keep:**
- `val/flip_rate` and `batch/effective` (kept on request, although flat while `flip_test` is off and without a
  batch schedule);
- `train/loss`, `train/loss_ema`, `train/loss/single`, `train/loss/multi`, `train/softmax_entropy`;
- `lr/head`, `lr/block_11`, `lr/block_10`;
- `optim/grad_norm`, `optim/grad_clipped` (useful again once the clip is 20), `optim/amp_scale`;
- `val/pck@0.1`, `val/pck@0.1_ci95`, `val/pck@0.05`, `val/pck@0.15`, `val/loss`, `val/loss/single`,
  `val/loss/multi`, `val/pck@0.1/{single, multi}`;
- the error-type bar plot, the `per_group` table, the distance histogram and the worst-prediction images.

PCK by size stays: it is how the small-object drop showed up.

**Add:**

| key | what |
|---|---|
| `train/softmax_entropy_ema` | 0.95 EMA of the entropy, like `train/loss_ema` (1.7) |
| `train/embed_var_batch` | 1 − ‖mean‖² of the batch's query descriptors (1.11) |
| `train/embed_var_cells` | 1 − ‖mean‖² over the cells of each target map, averaged over the batch (1.11) |
| `train/peak_mass` | probability inside a 5 × 5-cell window around the most likely cell: one peak (≈ 1) or several (1.4) |

The worst-prediction images stay.
## 2.3 Training changes

- **First, before any training: score `best.pt` with the better inference settings** (2.1, item 7).
  `--run` reuses the run's saved `config.json`, not `BASE` (RUN.md), so:
  1. In `<output_dir>/20261104-155818_train/config.json` (on Drive), set `"soft_window": 11` and
     `"soft_temperature": 0.02`, and keep `"flip_test": false`. Keep a copy of the original file.
  2. Run `python run.py --mode evaluate --run 20261104-155818_train`. Expect val PCK@0.10 ≈ 66.0, with the full
     per-group and error-type report.
  3. If it confirms, run `python run.py --mode submit --run 20261104-155818_train` for a submission from these
     settings.
  4. Set the same two values in `BASE`, so run 2 validates each epoch (and picks `best.pt`) with them.
- **A slower warmup and lower learning rates**, against the entropy overshoot (1.8):
  - warmup 10% → 20% of the updates (127 updates, one epoch at 5 epochs);
  - **head learning rate halved, 1e-4 → 5e-5;**
  - **backbone learning rate halved, 1e-5 → 5e-6.** Block 11 gets 5e-6, block 10 3.75e-6; the head stays at
    10× the backbone.

  Risk: half the learning rate over 5 epochs instead of 7 is about a third of run 1's total learning-rate budget.
  Under-training would show as val PCK still rising at epoch 4 (1.15). Then raise the epochs, not the learning
  rate.
- **Clipping as a spike guard:** `grad_clip` 1 → 20 (1.9).
- **Very light augmentation** (1.14), lighter than first proposed here:
  - only 10% of the images (`aug_image_prob = 0.10`) go through crop / rotation / colour / grayscale / blur;
    measured: 9.7% of the images change (run 1: 96%). The pair flip (0.5) is unchanged;
  - a crop keeps at least 90% of the side, rotation at most ± 10°;
  - lower label jitter, for the small objects (1.12).
- **One trained block:** `unfreeze_blocks` 2 → 1 (block 11 + final LayerNorm + head, 7.68M parameters).
- **5 epochs, patience 2** (1.15).

## 2.4 Change list for `BASE` in `src/config.py` (applied)

| setting | now | proposed | section |
|---|---|---|---|
| `epochs` | 7 | 5 | 1.15 |
| `early_stop_patience` | 3 | 2 | 1.15 |
| `warmup_pct` | 0.10 | 0.20 | 1.8 |
| `soft_window` | 5 | 11 | 1.4, 2.1 |
| `soft_temperature` | 0.04 | 0.02 | 1.4, 2.1 |
| `flip_test` | False | False (unchanged) | 2.1 |
| `lr_head` | 1e-4 | 5e-5 | 1.8 |
| `lr_backbone` | 1e-5 | 5e-6 | 1.8, 2.3 |
| `grad_clip` | 1.0 | 20.0 | 1.9 |
| `unfreeze_blocks` | 2 | 1 | 1.1 |
| `aug_image_prob` (new) | (1.0) | 0.10 | 1.14 |
| `aug_rotation_deg` / `aug_rotation_prob` | 15 / 0.5 | 10 / 0.5 | 1.14 |
| `aug_crop_scale` / `aug_crop_prob` | [0.7, 1.0] / 0.5 | [0.9, 1.0] / 0.5 | 1.14 |
| `aug_color_jitter` / `aug_hue_jitter` / `aug_color_prob` | 0.3 / 0.05 / 0.8 | 0.2 / 0.02 / 0.8 | 1.14 |
| `aug_grayscale_prob` / `aug_blur_prob` | 0.1 / 0.1 | 0.05 / 0.05 | 1.14 |
| `flip_augment_prob` | 0.5 | 0.5 (unchanged) | 1.14 |
| `label_jitter` | 0.5 | 0.25 | 1.12 |
Unchanged: `effective_batch = 16`, `loss_temperature = 0.03`, `label_sigma = 1.0`. New: `peak_mass_window = 5`.

The two matching settings also change the **submission** made without `--run` (the frozen baseline). On the frozen
model, window 11 with T = 0.02 scored 41.1 against 40.4 for window 5 with T = 0.04, so that is not a loss either.

Code changes:

- `src/utils/wandb_log.py`: no system stats, no console capture.
- `src/train.py`: the log keys of 2.2, the entropy EMA, the two embedding-variance values, the peak mass
  (computed in `src/models/loss.py`).
- `src/diagnostics.py::log_evaluation`: drop the deleted `val/*` keys; `n_queries` / `n_pairs` go to the run summary.
- `src/augment.py`: the `aug_image_prob` gate.

**Caveat:** run 2 changes several settings at once, so a difference with run 1 (62.9) cannot be credited to one of
them. That is acceptable here, because every change makes training *gentler* and none is expected to cost PCK. If
run 2 is worse by more than the interval, the first thing to undo is the learning rates (back to 1e-4 / 1e-5), then
the augmentation. Compare run 2 with run 1 **at the same inference settings** (66.0 with window 11 and T = 0.02,
not 62.9).

## 2.5 Next runs

0. **Evaluate `best.pt` of run 1 with `soft_window = 11`, `soft_temperature = 0.02`, no flip test** (steps in 2.3).
   No GPU training needed: a larger immediate gain (+3.1) than the last two training epochs together (+0.9).
1. **Run 2** with the change list above. Compare val PCK (overall and small objects), the entropy peak, and
   `grad_clipped` (≈ 0 after the warmup).
2. **The resolution sweep** (756 / 840) on the T4 with the fine-tuned model. Near misses and small objects are
   the largest remaining errors, and a finer grid is the direct fix.
3. **`flip_test`** on the fine-tuned model: does the frozen sweep's +3.1 survive fine-tuning?
4. **Find out why run 1 stopped at update 788** (Colab timeout or a manual stop?). `--resume` would have finished
   epoch 5
