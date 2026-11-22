"""Multi-model similarity-map ensemble (with shared 2-stage object-centric zoom).

Combines two or more trained CorrespondenceModel checkpoints (e.g. LoRA + Finetune) by
averaging their cosine-similarity maps before locate(). When `object_zoom` is True:
1. Stage 1 averages all models' similarity maps over `source_xy` context points to pick one
   shared, lower-variance Stage-2 crop box (sbox, tbox).
2. Stage 2 extracts zoomed crop features for each model on (s_crop, t_crop), averages their
   Stage-2 similarity maps, and runs locate() (and optional `constellation_align`).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.transforms import to_original, to_resized
from src.models.constellation import apply_similarity_2d, load_pair_source_xy, robust_constellation_fit
from src.models.matching import cell_centers, choose_source, locate, sample_point_features
from src.models.zoom import _sample_stage1_into_crop, _upsample_trg, bbox_crop


class EnsembleModel(nn.Module):
    """Weighted average of multiple CorrespondenceModel instances in similarity-map space."""

    def __init__(self, models, cfg, weights=None):
        super().__init__()
        if len(models) < 2:
            raise ValueError("EnsembleModel requires at least 2 models.")
        self.models = nn.ModuleList(models)
        self.cfg = cfg
        if weights is None:
            w = [1.0 / len(models)] * len(models)
        else:
            total = float(sum(weights))
            w = [float(x) / total for x in weights]
        self.weights = w

    def new_cache(self):
        return self.models[0].new_cache()

    @torch.no_grad()
    def image_features(self, img, flip=False, cache=None, key=None):
        """Delegate to model 0 for validation_loss logging compatibility."""
        return self.models[0].image_features(img, flip=flip, cache=cache, key=(0, key))

    def _model_sim_maps(self, idx, src_feats, trg_feats, pts_orig, src_wh, cfg, flipped=False):
        """[K, H, W] cosine similarity maps for points [K, 2] on one member model."""
        size = int(cfg["image_size"])
        xy = to_resized(pts_orig, src_wh, size).to(src_feats.device)
        if flipped:
            xy[:, 0] = size - xy[:, 0]
        q = sample_point_features(src_feats, xy, size)  # [K, C]
        up_trg = _upsample_trg(trg_feats, cfg)
        return torch.einsum("kc,chw->khw", q, up_trg)

    def _combine_sims(self, sims):
        """Weighted average of [K, H_i, W_i] similarity maps (resizing if grids differ)."""
        target_hw = max(s.shape[-2:] for s in sims)
        out = None
        for w, sim in zip(self.weights, sims):
            if sim.shape[-2:] != target_hw:
                sim = F.interpolate(sim[None], size=target_hw, mode="bilinear", align_corners=False)[0]
            out = w * sim if out is None else (out + w * sim)
        return out

    @torch.no_grad()
    def _pair_stage_features(self, src_img, trg_img, src_key, trg_key, cfg, cache):
        """Extract (or reuse) (src_feats, trg_feats, flipped) for every member model on one image pair."""
        per_model = []
        flipped = False
        if cfg["flip_test"]:
            # Decide flip consistently from model 0 (or ensemble alignment) so all members share coordinates
            s0 = self.models[0].image_features(src_img, cache=cache, key=(0, src_key))
            t0 = self.models[0].image_features(trg_img, cache=cache, key=(0, trg_key))
            s0, t0_raw = self.models[0].pair_features(s0, t0)
            s0_f = self.models[0].image_features(src_img, flip=True, cache=cache, key=(0, src_key))
            s0_f, _ = self.models[0].pair_features(s0_f, t0)
            _, flipped = choose_source(s0, s0_f, t0_raw)

        for idx, m in enumerate(self.models):
            sf = m.image_features(src_img, flip=flipped, cache=cache, key=(idx, src_key))
            tf = m.image_features(trg_img, cache=cache, key=(idx, trg_key))
            sf, tf = m.pair_features(sf, tf)
            per_model.append((sf, tf))
        return per_model, flipped

    @torch.no_grad()
    def predict(self, sample, cfg=None, cache=None):
        cfg = cfg or self.cfg
        size = int(cfg["image_size"])
        src_wh, trg_wh = sample["src_img"].size, sample["trg_img"].size
        stage1_feats, flipped = self._pair_stage_features(
            sample["src_img"], sample["trg_img"], sample["src_image"], sample["trg_image"], cfg, cache
        )

        if not cfg.get("object_zoom"):
            sims = [self._model_sim_maps(i, sf, tf, sample["src_xy"][None], src_wh, cfg, flipped)
                    for i, (sf, tf) in enumerate(stage1_feats)]
            sim = self._combine_sims(sims)
            pred = locate(sim, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
            return to_original(pred.cpu(), trg_wh, size).numpy()[0], flipped

        # --- Shared 2-Stage Object-Centric Zoom ---
        pair_key = ("ens_zoom", str(sample["pair_id"]), sample["src_image"], sample["trg_image"], size, bool(flipped))
        cached = cache.get(pair_key) if cache is not None else None
        if cached is not None:
            sbox, tbox, stage2_feats, ctx_xy = cached
        else:
            ctx_xy = load_pair_source_xy(cfg, sample["pair_id"], fallback_xy=sample["src_xy"][None])
            sims1_ctx = [self._model_sim_maps(i, sf, tf, ctx_xy, src_wh, cfg, flipped)
                         for i, (sf, tf) in enumerate(stage1_feats)]
            sim1_ctx = self._combine_sims(sims1_ctx)
            coarse_res = locate(sim1_ctx, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
            coarse_ctx = to_original(coarse_res.cpu(), trg_wh, size).numpy()

            max_area = float(cfg.get("zoom_max_area", 1.0))
            s_crop, sbox = bbox_crop(sample["src_img"], ctx_xy, cfg["zoom_margin"], cfg["zoom_min_frac"],
                                     cfg["zoom_pct_low"], cfg["zoom_pct_high"], max_area=max_area)
            t_crop, tbox = bbox_crop(sample["trg_img"], coarse_ctx, cfg["zoom_margin"], cfg["zoom_min_frac"],
                                     cfg["zoom_pct_low"], cfg["zoom_pct_high"], max_area=max_area)
            full_src = sbox == (0, 0, src_wh[0], src_wh[1])
            full_trg = tbox == (0, 0, trg_wh[0], trg_wh[1])
            if full_src and full_trg:
                stage2_feats = stage1_feats
            else:
                stage2_feats = []
                for idx, m in enumerate(self.models):
                    sf2 = stage1_feats[idx][0] if full_src else m.image_features(
                        s_crop, flip=flipped, cache=cache, key=(idx, "zoom_src", str(sample["pair_id"]), sbox, size))
                    tf2 = stage1_feats[idx][1] if full_trg else m.image_features(
                        t_crop, cache=cache, key=(idx, "zoom_trg", str(sample["pair_id"]), tbox, size))
                    sf2, tf2 = m.pair_features(sf2, tf2)
                    stage2_feats.append((sf2, tf2))
            if cache is not None:
                cache.put(pair_key, (sbox, tbox, stage2_feats, ctx_xy))

        sc_wh = (sbox[2] - sbox[0], sbox[3] - sbox[1])
        tc_wh = (tbox[2] - tbox[0], tbox[3] - tbox[1])
        q_in = np.asarray(sample["src_xy"], dtype=np.float32) - np.array(sbox[:2], dtype=np.float32)
        inside = (0.0 <= q_in[0] < sc_wh[0]) and (0.0 <= q_in[1] < sc_wh[1])
        if not inside:
            sims1 = [self._model_sim_maps(i, sf, tf, sample["src_xy"][None], src_wh, cfg, flipped)
                     for i, (sf, tf) in enumerate(stage1_feats)]
            sim1 = self._combine_sims(sims1)
            pred = locate(sim1, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
            return to_original(pred.cpu(), trg_wh, size).numpy()[0], flipped

        sims2 = [self._model_sim_maps(i, sf2, tf2, q_in[None], sc_wh, cfg, flipped)
                 for i, (sf2, tf2) in enumerate(stage2_feats)]
        sim2 = self._combine_sims(sims2)

        w1 = float(cfg.get("zoom_stage1_weight", 0.0))
        is_full = (sbox == (0, 0, src_wh[0], src_wh[1])) and (tbox == (0, 0, trg_wh[0], trg_wh[1]))
        if not is_full and w1 > 0.0:
            sims1 = [self._model_sim_maps(i, sf, tf, sample["src_xy"][None], src_wh, cfg, flipped)
                     for i, (sf, tf) in enumerate(stage1_feats)]
            sim1 = self._combine_sims(sims1)
            if float(sim1.max()) > float(sim2.max()) + 0.05:
                pred1 = locate(sim1, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
                return to_original(pred1.cpu(), trg_wh, size).numpy()[0], flipped
            sim1_crop = _sample_stage1_into_crop(sim1, tbox, trg_wh, sim2.shape[1:], size)
            sim2 = (1.0 - w1) * sim2 + w1 * sim1_crop

        if cfg.get("constellation_align"):
            ctx_in = np.asarray(ctx_xy, dtype=np.float32) - np.array(sbox[:2], dtype=np.float32)
            keep = (ctx_in[:, 0] >= 0.0) & (ctx_in[:, 0] < sc_wh[0]) & (ctx_in[:, 1] >= 0.0) & (ctx_in[:, 1] < sc_wh[1])
            ctx_valid = ctx_in[keep] if keep.any() else q_in[None]
            combined = np.vstack([q_in[None], ctx_valid])
            sims_all = self._combine_sims([
                self._model_sim_maps(i, sf2, tf2, combined, sc_wh, cfg, flipped)
                for i, (sf2, tf2) in enumerate(stage2_feats)
            ])
            raw_res = locate(sims_all, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
            conf = sims_all.flatten(1).max(dim=-1).values.detach().cpu().numpy()
            src_res = to_resized(combined, sc_wh, size).numpy()
            if flipped:
                src_res[:, 0] = size - src_res[:, 0]
            transform, _, _ = robust_constellation_fit(src_res, raw_res.detach().cpu().numpy(), conf, size, cfg)
            prior_weight = float(cfg["constellation_prior_weight"])
            if transform is not None and prior_weight > 0:
                exp_res = torch.as_tensor(apply_similarity_2d(src_res[:1], transform), dtype=torch.float32, device=sim2.device)
                _, h, w = sim2.shape
                cx, cy = cell_centers(w, size, sim2.device), cell_centers(h, size, sim2.device)
                sigma = float(cfg["constellation_sigma_px"])
                bonus = torch.exp(-((cx[None, None, :] - exp_res[:, 0, None, None]) ** 2
                                    + (cy[None, :, None] - exp_res[:, 1, None, None]) ** 2) / (2.0 * sigma ** 2))
                sim2 = sim2 + prior_weight * bonus

        pred2 = locate(sim2, size, cfg["matcher"], cfg["soft_window"], cfg["soft_temperature"])
        pred_xy = to_original(pred2.cpu(), tc_wh, size).numpy()[0]
        return pred_xy + np.array(tbox[:2], dtype=np.float32), flipped
