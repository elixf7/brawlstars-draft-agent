"""Training the antisymmetric field-aware model.

No flip augmentation: symmetry is structural now, so each game is one row
instead of two. That halves the training set without losing anything the
augmentation was there to teach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from bsdraft.data.sources import TEAM1_BRAWLER_COLS, TEAM2_BRAWLER_COLS
from bsdraft.fm.ffm import AntisymmetricFFM, FFMInference


def build_vocabularies(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    cols = TEAM1_BRAWLER_COLS + TEAM2_BRAWLER_COLS
    brawlers = pd.unique(df[cols].to_numpy().ravel())
    vocab = sorted(b for b in brawlers if isinstance(b, str))
    return vocab, sorted(df["map"].dropna().unique()), sorted(df["mode"].dropna().unique())


def encode(df: pd.DataFrame, vocab, maps, modes) -> dict[str, np.ndarray]:
    """Structured index arrays, rather than a generic sparse matrix.

    Unknown brawlers, maps or modes map to index 0 — new ones ship every season
    and a crash at inference time would be worse than a mild bias.
    """
    vi = {b: i for i, b in enumerate(vocab)}
    mi = {m: i for i, m in enumerate(maps)}
    di = {m: i for i, m in enumerate(modes)}
    to_idx = lambda s, lut: s.map(lut).fillna(0).astype(np.int64).to_numpy()  # noqa: E731

    # pandas can hand back read-only views, which torch refuses to wrap without
    # warning. ascontiguousarray preserves the flag, so copy when it is unset.
    return {
        key: arr if arr.flags.writeable else arr.copy()
        for key, arr in {
            "t1": np.stack([to_idx(df[c], vi) for c in TEAM1_BRAWLER_COLS], axis=1),
            "t2": np.stack([to_idx(df[c], vi) for c in TEAM2_BRAWLER_COLS], axis=1),
            "map": to_idx(df["map"], mi),
            "mode": to_idx(df["mode"], di),
            "skill": df["skill_ns"].fillna(0.0).to_numpy(dtype=np.float32),
            "y": df["team1_wins"].to_numpy(dtype=np.float32),
        }.items()
    }


def _metrics(model, enc, device, batch=32768) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(enc["y"]), batch):
            sl = slice(i, i + batch)
            outs.append(torch.sigmoid(model(
                torch.from_numpy(enc["t1"][sl]).to(device),
                torch.from_numpy(enc["t2"][sl]).to(device),
                torch.from_numpy(enc["map"][sl]).to(device),
                torch.from_numpy(enc["mode"][sl]).to(device),
                torch.from_numpy(enc["skill"][sl]).to(device),
            )).cpu().numpy())
    p = np.clip(np.concatenate(outs), 1e-7, 1 - 1e-7)
    y = enc["y"]
    return {
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }, p


def train_ffm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    k: int = 32,
    lr: float = 3e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 8192,
    max_epochs: int = 40,
    patience: int = 5,
    model_path: Path | None = None,
    verbose: bool = True,
) -> FFMInference:
    vocab, maps, modes = build_vocabularies(train_df)
    tr, va = encode(train_df, vocab, maps, modes), encode(val_df, vocab, maps, modes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AntisymmetricFFM(len(vocab), len(maps), len(modes), k=k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    n = len(tr["y"])
    tensors = {key: torch.from_numpy(val).to(device) for key, val in tr.items()}
    best, best_state, stale, history = float("inf"), None, 0, []

    if verbose:
        params = sum(p.numel() for p in model.parameters())
        print(f"AntisymmetricFFM  k={k}  {params:,} parameters  "
              f"{len(vocab)} brawlers / {len(maps)} maps / {len(modes)} modes")
        print(f"{n:,} train rows (no augmentation), {len(va['y']):,} val\n")
        print("Epoch  Train Loss    Val Loss      AUC  Status")
        print("-" * 54)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total, batches = 0.0, 0
        for i in range(0, n, batch_size):
            b = perm[i: i + batch_size]
            opt.zero_grad()
            loss = criterion(
                model(tensors["t1"][b], tensors["t2"][b], tensors["map"][b],
                      tensors["mode"][b], tensors["skill"][b]),
                tensors["y"][b],
            )
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1

        train_loss = total / batches
        val_metrics, _ = _metrics(model, va, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_logloss": val_metrics["logloss"], "val_auc": val_metrics["auc"]})

        improved = val_metrics["logloss"] < best - 1e-5
        if improved:
            best, stale = val_metrics["logloss"], 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose:
            mark = "✓ best" if improved else f"({stale}/{patience})"
            print(f"{epoch:5d} {train_loss:11.4f} {val_metrics['logloss']:11.4f}"
                  f" {val_metrics['auc']:8.4f}  {mark}")
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final, _ = _metrics(model, va, device)
    if verbose:
        print(f"\n  Log-loss : {final['logloss']:.4f}")
        print(f"  AUC-ROC  : {final['auc']:.4f}")
        print(f"  Brier    : {final['brier']:.4f}")

    inference = FFMInference.from_model(
        model, vocab, maps, modes,
        val_logloss=final["logloss"], val_auc=final["auc"], val_brier=final["brier"],
        n_train=n, n_val=len(va["y"]), train_history=history,
    )
    if model_path is not None:
        import pickle
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(inference, f)
    return inference


def predict_df(inf: FFMInference, df: pd.DataFrame) -> np.ndarray:
    enc = encode(df, inf.vocab, inf.maps, inf.modes)
    return inf.predict(enc["t1"], enc["t2"], enc["map"], enc["mode"], enc["skill"])


__all__ = ["build_vocabularies", "encode", "predict_df", "train_ffm"]
