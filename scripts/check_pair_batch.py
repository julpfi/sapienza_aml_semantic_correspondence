"""CPU checks of pair-batched training (src/pair_batch.py, augment_pair, pair_matching_loss, src/train.py).

    python scripts/check_pair_batch.py

Prints PASS/FAIL per check and exits non-zero on a failure. Uses the local training data, no network,
no model download: the v2-equivalence checks run src/train.py and its HEAD version (git show) on a tiny
random convolution in place of DINOv2.
"""

import ast
import importlib.util
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import QueryDataset, build_dataset, collate_samples, read_queries  # noqa: E402
from src.augment import _margins, augment_pair, augment_sample  # noqa: E402
from src.config import get_config  # noqa: E402
from src.models.loss import hard_negative_cells, matching_loss, pair_matching_loss  # noqa: E402
from src.pair_batch import PairDataset  # noqa: E402

FAILED = []


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def base_cfg(**overrides):
    cfg = get_config()
    cfg.update(wandb_mode="disabled", **overrides)
    return cfg


def load_head_module(rel_path, name, folder):
    """Import the HEAD version of a repo file under another module name."""
    source = subprocess.run(["git", "show", f"HEAD:{rel_path}"], cwd=REPO_ROOT, check=True,
                            capture_output=True, text=True).stdout
    path = Path(folder) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


# ---------------------------------------------------------------------------
def check_pair_dataset():
    cfg = base_cfg(max_train_queries=40)
    df = read_queries(cfg, "train")
    queries, pairs = QueryDataset(df, cfg), PairDataset(df, cfg)
    by_id = {queries[i]["query_id"]: queries[i] for i in range(len(queries))}
    seen, ok, detail = [], True, ""
    for j in range(len(pairs)):
        item = pairs[j]
        k_count = len(item["query_ids"])
        ok &= len(item["src_xys"]) == k_count == len(item["trg_xys"]) == len(item["scale"])
        for k, query_id in enumerate(item["query_ids"]):
            s = by_id[query_id]
            seen.append(query_id)
            same = (s["pair_id"] == item["pair_id"] and s["src_image"] == item["src_image"]
                    and s["trg_image"] == item["trg_image"] and s["category"] == item["category"]
                    and np.array_equal(s["src_xy"], item["src_xys"][k])
                    and np.array_equal(s["trg_xys"], item["trg_xys"][k]) and s["scale"] == item["scale"][k]
                    and s["src_img"].size == item["src_img"].size and s["trg_img"].size == item["trg_img"].size)
            if not same:
                ok, detail = False, f"mismatch on {query_id}"
    first_order = list(dict.fromkeys(by_id[q]["pair_id"] for q in df[cfg["csv_columns"]["query_id"]].astype(str)))
    ok &= sorted(seen) == sorted(by_id) and len(seen) == len(set(seen))
    ok &= pairs.pair_ids == first_order
    report("PairDataset == QueryDataset on the first 40 training rows", ok,
           detail or f"{len(pairs)} pairs, {len(seen)} queries, pair order = first appearance")


# ---------------------------------------------------------------------------
def random_problem(gen, n_queries, size, c=32, n=16):
    src = F.normalize(torch.randn(c, n, n, generator=gen), dim=0)
    trg = F.normalize(torch.randn(c, n, n, generator=gen), dim=0)
    src_xys = torch.rand(n_queries, 2, generator=gen) * size
    trg_xys = [torch.rand(int(torch.randint(1, 4, (1,), generator=gen)), 2, generator=gen) * size
               for _ in range(n_queries)]
    return src, trg, src_xys, trg_xys


def check_pair_loss_equals_matching_loss():
    gen = torch.Generator().manual_seed(0)
    for loss in ("soft_ce", "marginal"):
        cfg = base_cfg(image_size=224, loss=loss, hard_neg_weight=1.0)
        worst = 0.0
        for _ in range(5):
            src, trg, src_xys, trg_xys = random_problem(gen, 6, cfg["image_size"])
            pair = pair_matching_loss(src, trg, src_xys, trg_xys, cfg, temperature=0.05)
            for k in range(len(src_xys)):
                single = matching_loss(src, trg, src_xys[k][None], trg_xys[k], cfg, temperature=0.05)
                worst = max(worst, max(abs(float(a[k]) - float(b)) for a, b in zip(pair, single)))
        report(f"pair_matching_loss == matching_loss per query, weight 1 ({loss})", worst < 1e-5,
               f"max abs diff {worst:.2e}")


def check_matching_loss_unchanged(head_loss):
    gen = torch.Generator().manual_seed(1)
    for loss in ("soft_ce", "marginal"):
        cfg = base_cfg(image_size=224, loss=loss)
        worst = 0.0
        for _ in range(5):
            src, trg, src_xys, trg_xys = random_problem(gen, 1, cfg["image_size"])
            new = matching_loss(src, trg, src_xys, trg_xys[0], cfg)
            old = head_loss.matching_loss(src, trg, src_xys, trg_xys[0], cfg)
            worst = max(worst, max(abs(float(a) - float(b)) for a, b in zip(new, old)))
        report(f"matching_loss identical to HEAD ({loss})", worst == 0.0, f"max abs diff {worst:.2e}")


# ---------------------------------------------------------------------------
def cell_xy(row, col, cell):
    return [(col + 0.5) * cell, (row + 0.5) * cell]


def check_hard_negatives():
    cfg = base_cfg(image_size=224, hard_neg_weight=3.0, hard_neg_radius=1.0, hard_neg_exclusion=2.0)
    n, cell = 16, 224 / 16
    idx = lambda row, col: row * n + col  # noqa: E731
    # Query 0: one target at cell (3, 3). Query 1: targets at (12, 12) and (4, 4) (next to query 0's).
    trg_xys = [torch.tensor([cell_xy(3, 3, cell)]), torch.tensor([cell_xy(12, 12, cell), cell_xy(4, 4, cell)])]
    mask = hard_negative_cells(trg_xys, n, n, cfg, "cpu")
    # (5, 4): 1 cell from query 1's (4, 4), 2.2 cells from query 0's own (3, 3) -> a negative of query 0.
    near_other = all(bool(mask[0, idx(r, c)]) for r, c in ((12, 12), (12, 13), (5, 4)))
    near_other &= bool(mask[1, idx(3, 2)])  # 1 cell from query 0's target, 2.2 from query 1's (4, 4)
    # (4, 3): 1 cell from query 1's (4, 4) but also 1 cell from query 0's own target -> excluded.
    own_free = not any(bool(mask[0, idx(r, c)]) for r, c in ((3, 3), (4, 4), (4, 3)))
    own_free &= not any(bool(mask[1, idx(r, c)]) for r, c in ((4, 4), (3, 3), (12, 12), (12, 11)))
    far = not bool(mask[0, idx(8, 8)] or mask[1, idx(8, 8)])
    report("hard_negative_cells: cells near another query's target are True", near_other)
    report("hard_negative_cells: cells near own targets (exclusion) and far cells are False", own_free and far,
           f"negatives per query {mask.sum(-1).tolist()}")
    single = hard_negative_cells(trg_xys[:1], n, n, cfg, "cpu")
    report("hard_negative_cells: one query in the pair -> empty mask", not bool(single.any()))

    # Query 0 whose feature matches the cell of query 1's target (12, 12): a prediction on a negative.
    gen = torch.Generator().manual_seed(2)
    trg = F.normalize(torch.randn(32, n, n, generator=gen), dim=0)
    src = F.normalize(torch.randn(32, n, n, generator=gen), dim=0)
    src[:, 3, 3] = trg[:, 12, 12]
    src_xys = torch.tensor([cell_xy(3, 3, cell), cell_xy(8, 8, cell)])
    for loss in ("soft_ce", "marginal"):
        losses = {}
        for weight in (1.0, 3.0):
            c = dict(cfg, loss=loss, hard_neg_weight=weight)
            losses[weight] = float(pair_matching_loss(src, trg, src_xys, trg_xys, c, temperature=0.05)[0][0])
        report(f"hard negatives raise the loss of a query predicted on a negative ({loss})",
               losses[3.0] > losses[1.0], f"weight 1: {losses[1.0]:.3f}, weight 3: {losses[3.0]:.3f}")


# ---------------------------------------------------------------------------
def check_augment_pair():
    cfg = base_cfg(max_train_queries=40)
    pairs = PairDataset(read_queries(cfg, "train"), cfg)
    items = [pairs[j] for j in range(len(pairs))]
    random.seed(cfg["seed"])
    aligned, edge_violations, empty, dropped, total = True, 0, 0, 0, 0
    for t in range(200):
        item = items[t % len(items)]
        a = augment_pair(item, cfg)
        k = len(a["query_ids"])
        total += len(item["query_ids"])
        dropped += len(item["query_ids"]) - k
        empty += k == 0
        aligned &= k == len(a["src_xys"]) == len(a["trg_xys"]) == len(a["scale"])
        aligned &= all(len(x) > 0 for x in a["trg_xys"]) and set(a["query_ids"]) <= set(item["query_ids"])
        for k_new, query_id in enumerate(a["query_ids"]):
            k_old = item["query_ids"].index(query_id)
            aligned &= a["scale"][k_new] == item["scale"][k_old] and len(a["trg_xys"][k_new]) <= len(item["trg_xys"][k_old])
        for orig_img, orig_xy, img, xy in ((item["src_img"], item["src_xys"], a["src_img"], a["src_xys"]),
                                           (item["trg_img"], np.concatenate(item["trg_xys"]), a["trg_img"],
                                            np.concatenate(a["trg_xys"]) if a["trg_xys"] else np.zeros((0, 2)))):
            w = orig_img.size[0]
            unmoved = np.concatenate([orig_xy, np.stack([w - orig_xy[:, 0], orig_xy[:, 1]], 1)])  # or flipped
            m = _margins(img.size, cfg)
            for point in xy:
                inside = ((point >= m - 1e-6) & (point <= np.asarray(img.size) - m + 1e-6)).all()
                if not inside and np.abs(unmoved - point).max(axis=1).min() >= 1e-3:
                    edge_violations += 1
    report("augment_pair: per-query lists aligned, >= 1 query and >= 1 target per query kept",
           aligned and empty == 0, f"{dropped} of {total} queries dropped in 200 augmentations")
    report("augment_pair: edge rule (moved points stay aug_edge_margin inside)", edge_violations == 0,
           f"{edge_violations} violations")

    def run(seed):
        random.seed(seed)
        return [augment_pair(items[j], cfg) for j in range(min(6, len(items)))]

    first, second = run(123), run(123)
    same = all(x["query_ids"] == y["query_ids"] and np.array_equal(x["src_xys"], y["src_xys"])
               and all(np.array_equal(p, q) for p, q in zip(x["trg_xys"], y["trg_xys"]))
               and np.array_equal(np.asarray(x["src_img"]), np.asarray(y["src_img"]))
               and np.array_equal(np.asarray(x["trg_img"]), np.asarray(y["trg_img"]))
               and x["src_image"] == y["src_image"] for x, y in zip(first, second))
    report("augment_pair: same random.seed -> same result", same)


# ---------------------------------------------------------------------------
class FakeBackbone(nn.Module):
    """Stand-in for DinoBackbone: same freeze / unfreeze_last / blocks / layernorm interface."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.patch = nn.Conv2d(3, 32, kernel_size=14, stride=14)
        self.blocks = nn.ModuleList([nn.Conv2d(32, 32, 1) for _ in range(4)])
        self.layernorm = nn.GroupNorm(1, 32)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_last(self, n_blocks):
        for block in list(self.blocks)[-n_blocks:]:
            block.requires_grad_(True)
        self.layernorm.requires_grad_(True)

    def forward(self, x):
        x = self.patch(x)
        for block in self.blocks:
            x = x + block(x)
        return F.normalize(self.layernorm(x), dim=1)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.head = nn.Identity()

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, x):
        return self.head(self.backbone(x))


def loader_call(source):
    """The `loader = DataLoader(...)` statement of train() in a train.py source."""
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "DataLoader"):
            return node.value
    raise ValueError("no DataLoader call found")


def check_v2_equivalence(head_train):
    import src.train as new_train

    cfg = base_cfg(max_train_queries=48, image_size=224)
    new_source = (REPO_ROOT / "src" / "train.py").read_text()
    head_call, new_call = loader_call(head_train.SOURCE), loader_call(new_source)
    report("query-mode DataLoader call: same source as HEAD (ast)", ast.dump(head_call) == ast.dump(new_call),
           ast.unparse(new_call))

    model = FakeModel()
    groups = {}
    for name, module in (("head", head_train), ("new", new_train)):
        groups[name] = [(g["name"], g["lr"], sum(p.numel() for p in g["params"]))
                        for g in module.configure_trainable(model, cfg)]
    report("configure_trainable (finetune): same groups as HEAD", groups["head"] == groups["new"], str(groups["new"]))
    trainable = [p for p in model.parameters() if p.requires_grad]

    micro = {name: m.choose_micro_batch(model, cfg, [{"params": trainable}])[0]
             for name, m in (("head", head_train), ("new", new_train))}
    dataset = build_dataset(cfg, "train")
    n_batches = len(dataset) // micro["new"]
    schedule = {}
    for name, m in (("head", head_train), ("new", new_train)):
        plan, total = m.plan_updates(n_batches, micro[name], cfg)
        temperature_updates = m.updates_in_epochs(plan, cfg["loss_temperature_epochs"])
        schedule[name] = (micro[name], plan, total, m.updates_in_epochs(plan, cfg["warmup_epochs"]),
                          [m.loss_temperature_at(u, temperature_updates, cfg) for u in range(total)])
    report("micro-batch, update plan, warmup and temperature schedule: same as HEAD",
           schedule["head"] == schedule["new"], f"micro {micro['new']}, plan {schedule['new'][1]}")

    ids, losses = {}, {}
    for name, module, call in (("head", head_train, head_call), ("new", new_train, new_call)):
        ids[name], losses[name] = [], []
        for epoch in (0, 1):
            namespace = {"DataLoader": torch.utils.data.DataLoader, "dataset": dataset, "micro": micro[name],
                         "cfg": cfg, "collate_samples": collate_samples, "torch": torch, "epoch": epoch}
            loader = eval(compile(ast.Expression(call), "<loader>", "eval"), namespace)
            for i, samples in enumerate(loader):
                if i == 3:
                    break
                ids[name].append([s["query_id"] for s in samples])
                random.seed(1000 * epoch + i)
                torch.manual_seed(1000 * epoch + i)
                samples = [augment_sample(s, cfg) for s in samples]
                stats = module.batch_loss(model, samples, cfg, None, temperature=module.loss_temperature_at(
                    i, 10, cfg))
                losses[name].append(torch.stack(stats).detach())
    report("first 3 batches of epochs 0 and 1: same query ids as HEAD", ids["head"] == ids["new"],
           f"epoch 0 batch 0 = {ids['new'][0]}, epoch 1 batch 0 = {ids['new'][3]}")
    worst = max(float((a - b).abs().max()) for a, b in zip(losses["head"], losses["new"]))
    report("first 3 batches of epochs 0 and 1: same LOSS_STATS as HEAD", worst == 0.0, f"max abs diff {worst:.2e}")

    try:
        new_train.train(dict(cfg, hard_neg_weight=2.0), model, "check", Path(tempfile.gettempdir()))
        report("hard_neg_weight > 1 without pair_batch raises ValueError", False)
    except ValueError as err:
        report("hard_neg_weight > 1 without pair_batch raises ValueError", "pair_batch" in str(err))


def check_pair_batch_loss():
    """One update of pair mode by hand: pair_batch_loss on the fake model, stats per query."""
    import src.train as new_train

    cfg = base_cfg(max_train_queries=300, image_size=224, pair_batch=True, hard_neg_weight=2.0, loss="marginal")
    pairs = PairDataset(read_queries(cfg, "train"), cfg)
    largest = sorted(range(len(pairs)), key=lambda j: -len(pairs.rows[j]))[:2]  # pairs with several queries
    random.seed(0)
    items = [augment_pair(pairs[j], cfg) for j in largest]
    model = FakeModel()
    new_train.configure_trainable(model, cfg)
    stats, n_negatives = new_train.pair_batch_loss(model, items, cfg, None, temperature=0.05)
    n = sum(len(item["query_ids"]) for item in items)
    stats[0].sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    ok = all(len(v) == n for v in stats) and len(n_negatives) == n and has_grad and bool(torch.isfinite(stats[0]).all())
    report("pair_batch_loss: one value per query, finite, gradients reach the trained blocks", ok,
           f"{n} queries, hard_neg_cells mean {float(n_negatives.float().mean()):.1f}")
    for mode in (1, 2):
        cfg_q = dict(cfg, pair_batch=mode == 2, pair_effective_batch=4)
        expected = 4 if mode == 2 else cfg["effective_batch"]
        report(f"effective_batch_for with pair_batch={mode == 2}", new_train.effective_batch_for(0, cfg_q) == expected)


def main():
    with tempfile.TemporaryDirectory() as folder:
        head_train, source = load_head_module("src/train.py", "train_head", folder)
        head_train.SOURCE = source
        head_loss, _ = load_head_module("src/models/loss.py", "loss_head", folder)
        check_pair_dataset()
        check_matching_loss_unchanged(head_loss)
        check_pair_loss_equals_matching_loss()
        check_hard_negatives()
        check_augment_pair()
        check_v2_equivalence(head_train)
        check_pair_batch_loss()
    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: ' + '; '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
