# Data Improvement Brainstorm (Current `main` version audit)

## Current pipeline snapshot (what matters for data quality)

- Training currently uses a **photometric objective only**: masked L1 + global SSIM blend in `LossFunctions`.
- The code explicitly marks future work: "add more losses" and a note that SSIM is not masked.
- Dataset currently expects only canonical views from `VIEW_ORDER = [front, back, left, right]`.
- `data.num_views` effectively supports `1` or `4` in the current dataset indexing logic and README contract.
- THuman preprocessing renders those canonical views and already writes foreground masks from depth.

## Your 3 ideas: concrete upgrades

### 1) Add more robust losses

Good direction; this is likely the **highest-impact low-risk** change.

Recommended loss stack:

1. **Masked SSIM** (quick win)
   - You already mask L1 using non-black pixels; SSIM currently runs on full image.
   - Compute SSIM on foreground mask only, or blend with a masked variant.

2. **Perceptual loss (LPIPS / VGG features)**
   - Helps texture realism and reduces blur from pure L1 optimization.
   - Use moderate weight and keep mask-aware evaluation where possible.

3. **Alpha/silhouette consistency loss**
   - Since preprocessing can output foreground masks, compare rendered alpha/silhouette with GT mask.
   - This significantly improves boundary quality and floating artifacts.

4. **Multi-view consistency loss (geometry-aware)**
   - Enforce consistency of projected Gaussian attributes across views of same subject.
   - Start with color/opacity consistency for corresponding Gaussians.

5. **Gaussian regularizers**
   - Scale regularization to avoid over-large splats.
   - Opacity sparsity/entropy term to prevent "fog" artifacts.
   - Rotation/SH smoothness priors if instability appears.

### 2) Augment data (with "allowed" constraints)

For your setup (identity-specific, multi-view, camera-aware), augmentations must preserve geometry/camera semantics.

**Usually safe (apply identically across all views in same sample):**
- Color jitter (small brightness/contrast/saturation).
- Gamma / white-balance perturbation.
- JPEG compression / mild blur / sensor noise.
- Random background compositing if masks are available (subject fixed, background changed).

**Conditionally safe:**
- Small crop + resize only if intrinsics are updated consistently.
- Mild occluder cutout if used to simulate accessories/partial occlusion.

**Usually unsafe unless camera metadata is updated:**
- Horizontal flips (changes left/right semantics and handedness).
- Free rotations / perspective warps.
- Per-view independent strong color changes (breaks cross-view identity consistency).

### 3) Use more camera views in preprocessing

Strong idea for this model family.

- Today preprocessing and dataset contracts are centered on 4 canonical directions.
- Expanding to **8–16 azimuth views (+ optional top/down tilt)** can improve clothing/back-detail fidelity and reduce view extrapolation artifacts.
- Keep view naming + camera JSON generation systematic (e.g., `az000`, `az045`, ...).
- Increase views gradually (4 -> 8 -> 12) to monitor memory/training-time scaling.

## Additional ideas beyond your list

1. **Pose diversity expansion per identity**
   - If available, include multiple poses/frames per subject, not just one canonical pose.
   - Improves generalization of appearance under deformation.

2. **Quality filtering / curriculum**
   - Auto-score images for blur, texture completeness, mask quality.
   - Train first on high-quality subset, then full set.

3. **Hard-view mining**
   - Track per-view reconstruction error and oversample difficult views (often back/side).

4. **Subject balancing**
   - Ensure equal sampling over subjects to avoid overfitting frequent identities.

5. **Resolution curriculum**
   - Start at 256/512, finetune at 768/1024 when stable.
   - Faster convergence and fewer early training instabilities.

6. **Camera calibration sanity checks**
   - Add a preprocessing validation stage that reprojects SMPL-X vertices and checks 2D alignment error.
   - Miscalibrated cameras can dominate loss and hurt all subsequent experiments.

7. **Domain randomization for deployment gap**
   - If inference photos differ from synthetic THuman renders, inject realistic exposure/noise/background statistics.

## Suggested experiment order (fastest signal first)

1. Add masked SSIM + silhouette loss.
2. Add safe photometric augmentations (small jitter/noise/compression), synchronized across views.
3. Expand views 4 -> 8 while keeping same identities.
4. Add LPIPS and Gaussian regularizers.
5. Add hard-view mining and quality filtering.

## Success metrics to track

- Standard: L1, SSIM, LPIPS (validation).
- Boundary quality: mask IoU / contour F-score.
- Multi-view consistency: per-subject variance of reprojected color/opacity.
- Generalization: held-out viewpoints at unseen azimuths.
- Efficiency: GPU memory, step time, render FPS.

## Practical "next sprint" implementation checklist

- [ ] Add config weights for new loss terms (`weight_lpips`, `weight_silhouette`, `weight_scale_reg`, `weight_opacity_reg`).
- [ ] Extend preprocessing to save/verify masks and camera metadata per view.
- [ ] Add a view list config (`data.views`) instead of fixed `1` or `4` assumptions.
- [ ] Add augmentation module with **multi-view synchronized RNG**.
- [ ] Run ablation table: baseline / +loss / +aug / +views / combined.
