# Suggestions (cheapest first)

Context: run 1 = 62.9 val PCK@0.10 (66.0 with the 11×11 / T 0.02 window). Details in
[docs/REVIEW_finetuning.md](docs/REVIEW_finetuning.md); settings and charts in [docs/RUN.md](docs/RUN.md) §8–9.

1. dinov3 as better backbone
2. sd adapter change hyperparameter

# possible change the hyperparameter from the colab run

# Part 2 of the challenge: semantic correspondence + editing (brainstorm)

Source: `docs/AML Competition 26_27.pptx` (slides "The task", "Evaluation Metric", "The rules", "Kickstarter").
These are ideas to discuss, not decisions; nothing here is implemented.

## What the slides say

- **Input:** a source image with a source point, the same source image after an edit (edited image), and a target
  image. **Output:** the target image with the same edit applied to the corresponding part (the "?" on slide 11),
  plus the predicted target point.
- **Metric:** a weighted average of three scores: geometric (L2 distance between the predicted and the true target
  point), how good the edited part is, and how well the background is preserved. The exact formula is in the
  Kaggle competition.
- **Rules:** diffusion models are required. Non-diffusion architectures are not allowed, and neither is LLM
  annotation of the target image. Same groups as part 1.
- **Kickstarter 1:** SD 1.5 UNet + VAE extract source, edited and target features. The semantic correspondence maps
  the edit to the target, the edit direction is Δ = edited features − source features, and the output is
  target + W · Δ with a weight map W.
- **Kickstarter 2:** source − edited = ? Infer the prompt of the edit, then apply the edited prompt to the target
  image.

## Risk to check first

- "Non-diffusion architecture not allowed" may exclude DINOv2. Our part-1 matcher is DINOv2-L + SD. Ask the TAs
  whether DINOv2 may be used as part of the correspondence step. If not, part 2 needs an SD-only matcher (DIFT-style
  `sd_up_blocks` features alone). That mode does not exist yet: today the SD features are always fused with DINOv2.

## Pipeline sketch

1. **Correspondence (geometric score):** reuse the part-1 model to map the source point to the target point. Also
   produce a dense mapping around it (the similarity map, or the matches of every source cell), because the edit
   covers a region, not a point.
2. **Edit region:** the difference between the source and edited images (in pixels or in VAE latents), thresholded
   and restricted to an area around the source point, gives a source mask M_src. Warp M_src to the target with the
   dense correspondence to get M_trg; SD cross-attention maps can refine it (this stays diffusion-only, unlike SAM).
3. **Edit transfer**, two options from the kickstarters:
   - (a) Feature delta (kickstarter 1): invert the target with DDIM inversion (or null-text inversion) in SD 1.5,
     compute Δ at chosen UNet layers from the source / edited pair, warp Δ with the correspondence, and inject
     W · Δ during denoising, with W = M_trg (soft). Close to plug-and-play feature injection.
   - (b) Edit prompt (kickstarter 2): learn the edit as a text embedding from the source / edited pair (textual
     inversion or an "edit direction" in the CLIP text space of SD), then run SD inpainting on the target inside
     M_trg with that embedding. Captioning the target with an LLM is forbidden; check whether captioning the
     source / edited pair is allowed.
4. **Background preservation:** blended latent diffusion: at every denoising step, replace the latents outside
   M_trg with the inverted target latents, so only the edit region changes.
5. **Alternatives to try if time allows:** visual in-context methods that solve image analogies (A : A' :: B : ?)
   with an inpainting diffusion model on a 2 x 2 grid of the three images, and cross-image attention
   (keys / values from the edited source injected while denoising the target).

## Local evaluation (before Kaggle)

- Geometric: PCK on our validation split (part-1 code and `metric.py`).
- Background: PSNR / LPIPS outside M_trg between the output and the target.
- Edit quality: a proxy such as the CLIP similarity between the edited source crop and the output crop around the
  predicted point (evaluation only, not part of the method). Weight the three as the Kaggle metric does once its
  formula is known.

## Compute

SD 1.5 in fp16 fits a T4. DDIM inversion plus editing costs about 2 x 50 UNet steps per test image, so a submission
of a few hundred images takes on the order of an hour; cache the inversions.
