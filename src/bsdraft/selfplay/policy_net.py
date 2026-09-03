"""
policy_net.py — Part 4.2: Policy Network Architecture & Training

A single MLP policy network trained on self-play MCTS visit distributions.
One model handles both my-turn and opp-turn picks via a whose_turn input flag
(though in practice the flag is always 1.0 since records are stored from the
picker's perspective — it's included for completeness and future use).

Architecture
------------
  Input  : partial draft state encoded as dense float32 vector
           (FM feature space + whose_turn flag + pick_number one-hot) = n_features+7 dims
  Hidden : 512 → 256 → 128 (ReLU activations)
  Output : logits over all V vocabulary brawlers
           → masked softmax over available brawlers → policy prior

Training
--------
  Loss: cross-entropy proxy for KL divergence vs MCTS visit distribution
  Records split by is_primary_team_pick, separate mini-batches interleaved per epoch
  L2 regularisation weight_decay=1e-4, early stopping patience=5 on val cross-entropy
  Train/val split by game_id (not by pick) to prevent intra-game leakage

Integration
-----------
  PolicyInference.predict_prior(state, available) → probability vector over available
  Used as PUCT prior in _run_mcts when provided; falls back to counter-rate prior if None
  At opp-turn nodes, state is mirrored to picker's perspective before inference.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bsdraft.features.engineering import FeatureSchema
from bsdraft.mcts.state import DraftState

# src/bsdraft/<subpackage>/<module>.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
POLICY_NET_PATH = DATA_DIR / "policy_net.pkl"

# ── Feature layout constants ──────────────────────────────────────────────────

POLICY_EXTRA_FEATURES: int = 7
"""
Extra features appended after the FM feature block:
  [n_features + 0]     whose_turn flag (1.0 = mine, 0.0 = opp)
  [n_features + 1..6]  pick_number one-hot (6 positions for picks 0–5)

In practice whose_turn is always 1.0 (records are stored from picker's
perspective), but it is included for completeness and possible future use.
"""

# ── Architecture ─────────────────────────────────────────────────────────────

_HIDDEN: tuple[int, ...] = (512, 256, 128)

# ── Training hyperparameters ──────────────────────────────────────────────────

_LR: float = 1e-3
_WEIGHT_DECAY: float = 1e-4
_BATCH_SIZE: int = 512
_MAX_EPOCHS: int = 50
_PATIENCE: int = 5
_VAL_GAME_FRACTION: float = 0.2


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_partial_state(state: DraftState, schema: FeatureSchema) -> np.ndarray:
    """
    Encode a partial DraftState into a dense float32 vector of length
    schema.n_features + POLICY_EXTRA_FEATURES.

    Layout mirrors the FM's FeatureSchema (t1 = my_team, t2 = opp_team),
    extended with whose_turn flag and pick_number one-hot.

    State must be non-terminal (whose_turn is accessed).
    """
    n = schema.n_features + POLICY_EXTRA_FEATURES
    vec = np.zeros(n, dtype=np.float32)

    for b in state.my_team:
        idx = schema._vocab_idx.get(b)
        if idx is not None:
            vec[schema.t1_offset + idx] = 1.0

    for b in state.opp_team:
        idx = schema._vocab_idx.get(b)
        if idx is not None:
            vec[schema.t2_offset + idx] = 1.0

    m_idx = schema._map_idx.get(state.map_name)
    if m_idx is not None:
        vec[schema.map_offset + m_idx] = 1.0

    if schema.include_mode:
        mo_idx = schema._mode_idx.get(state.mode)
        if mo_idx is not None:
            vec[schema.mode_offset + mo_idx] = 1.0

    vec[schema.skill_offset] = float(state.skill_ns)

    # whose_turn flag
    vec[schema.n_features] = 1.0 if state.whose_turn == "mine" else 0.0

    # pick_number one-hot (0–5)
    vec[schema.n_features + 1 + min(state.pick_number, 5)] = 1.0

    return vec


def _picker_pov(state: DraftState) -> DraftState:
    """
    Return state from the picker's perspective (whose_turn == 'mine').

    If state.whose_turn == 'opp', swap my_team/opp_team and flip is_first_pick
    so the returned state has whose_turn == 'mine' at the same pick_number.
    This is the same mirroring used in self-play data generation (self_play.py),
    so inference is consistent with training.
    """
    if state.whose_turn == "opp":
        return DraftState(
            my_team=state.opp_team,
            opp_team=state.my_team,
            mode=state.mode,
            map_name=state.map_name,
            skill_ns=state.skill_ns,
            is_first_pick=not state.is_first_pick,
            bans=state.bans,
        )
    return state


# ── Model ─────────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """
    MLP policy network: partial draft state → brawler pick distribution.

    Input : dense float32 vector of length n_input (= schema.n_features + 7)
    Output: raw logits over all V vocabulary brawlers (before masking/softmax).

    Masking to available brawlers and normalization are done at inference time
    in PolicyInference.predict_prior(), not inside the forward pass, so the
    model can be trained on the full V-dimensional target once and masked at
    call time with different availability sets.
    """

    def __init__(
        self,
        n_input: int,
        n_vocab: int,
        hidden: tuple[int, ...] = _HIDDEN,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_input
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_vocab))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_input) → logits (B, n_vocab)"""
        return self.net(x)


# ── Inference wrapper ─────────────────────────────────────────────────────────

@dataclass
class PolicyInference:
    """
    Trained policy network wrapper for MCTS prior computation.

    Encodes a DraftState, runs the forward pass, masks unavailable brawlers,
    and returns a normalized probability distribution over available picks.

    At opponent-turn MCTS nodes (state.whose_turn == 'opp'), the state is
    automatically mirrored to the picker's perspective before encoding —
    consistent with how self-play records are stored (all whose_turn == 'mine').

    Usage
    -----
        policy = PolicyInference.load(data_dir / "policy_net.pkl")
        probs  = policy.predict_prior(state, available_brawlers)
        # probs[i] = probability of picking available_brawlers[i]
    """

    net: PolicyNet
    schema: FeatureSchema

    def predict_prior(
        self,
        state: DraftState,
        available: list[str],
    ) -> np.ndarray:
        """
        Return a normalized probability vector over `available` brawlers.

        Mirrors state to picker's perspective if whose_turn == 'opp'.
        All computations are done on CPU with torch.no_grad().

        Parameters
        ----------
        state     : non-terminal DraftState at the node being expanded
        available : available brawler names (from available_actions())

        Returns
        -------
        np.ndarray of shape (len(available),) summing to 1.0.
        Uniform fallback if no available brawler is in the vocabulary.
        """
        if not available:
            return np.array([], dtype=np.float32)

        eval_state = _picker_pov(state)
        vec = encode_partial_state(eval_state, self.schema)
        x = torch.from_numpy(vec).unsqueeze(0)  # (1, n_input)

        vocab_idx = self.schema._vocab_idx
        V = len(self.schema.vocab)

        avail_indices = [vocab_idx[b] for b in available if b in vocab_idx]
        if not avail_indices:
            return np.ones(len(available), dtype=np.float32) / len(available)

        with torch.no_grad():
            logits = self.net(x)[0]  # (V,)

        # Mask unavailable brawlers to -inf, then softmax over available only
        mask = torch.full((V,), float("-inf"))
        mask[avail_indices] = 0.0
        probs = torch.softmax(logits + mask, dim=0).numpy()

        result = np.array(
            [probs[vocab_idx[b]] if b in vocab_idx else 0.0 for b in available],
            dtype=np.float32,
        )
        s = result.sum()
        if s > 0:
            return result / s
        return np.ones(len(available), dtype=np.float32) / len(available)

    def save(self, path: Path = POLICY_NET_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"PolicyInference saved → {path}")

    @classmethod
    def load(cls, path: Path = POLICY_NET_PATH) -> PolicyInference:
        import importlib

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "__main__":
                    for mod_name in ("policy_net", "src.policy_net"):
                        try:
                            return getattr(importlib.import_module(mod_name), name)
                        except (ModuleNotFoundError, AttributeError):
                            pass
                return super().find_class(module, name)

        with open(path, "rb") as fh:
            return _Unpickler(fh).load()


# ── Training helpers ──────────────────────────────────────────────────────────

def _build_arrays(
    records: list,
    schema: FeatureSchema,
    V: int,
    n_input: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, availability_mask, targets) arrays for a list of SelfPlayRecords.

    X            : (N, n_input) float32 — encoded partial states
    avail_mask   : (N, V) bool — True where brawler is available
    targets      : (N, V) float32 — renormalized visit distribution (sums to 1.0 per row)
    """
    N = len(records)
    X = np.zeros((N, n_input), dtype=np.float32)
    avail_mask = np.zeros((N, V), dtype=bool)
    targets = np.zeros((N, V), dtype=np.float32)
    vocab_idx = schema._vocab_idx
    vocab = schema.vocab

    for i, rec in enumerate(records):
        X[i] = encode_partial_state(rec.state, schema)

        used = rec.state.my_team | rec.state.opp_team | rec.state.bans
        for j, b in enumerate(vocab):
            if b not in used:
                avail_mask[i, j] = True

        visit_mass = sum(rec.visit_dist.values())
        if visit_mass > 0:
            for b, frac in rec.visit_dist.items():
                idx = vocab_idx.get(b)
                if idx is not None and avail_mask[i, idx]:
                    targets[i, idx] = frac / visit_mass
        if targets[i].sum() == 0:
            n_avail = int(avail_mask[i].sum())
            if n_avail > 0:
                targets[i, avail_mask[i]] = 1.0 / n_avail

    return X, avail_mask, targets


def _kl_loss(
    net: PolicyNet,
    x: torch.Tensor,
    avail_mask: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy proxy for KL divergence vs MCTS visit distribution.

    KL(P || Q) = H(P, Q) - H(P). Since H(P) is constant w.r.t. model params,
    minimising KL ≡ minimising cross-entropy H(P, Q) = -Σ P * log Q.

    Unavailable brawlers are masked to -inf before log_softmax so they receive
    zero probability. To avoid 0 * (-inf) = NaN, log_probs are set to 0.0
    wherever the target is 0 before multiplying.
    """
    logits = net(x)
    logits = logits.masked_fill(~avail_mask, float("-inf"))
    log_probs = torch.log_softmax(logits, dim=-1)
    safe_log_probs = log_probs.masked_fill(targets == 0, 0.0)
    return -(targets * safe_log_probs).sum(dim=-1).mean()


def _make_batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


# ── Training ──────────────────────────────────────────────────────────────────

def train_policy(
    records: list,
    schema: FeatureSchema,
    *,
    lr: float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
    batch_size: int = _BATCH_SIZE,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
    val_game_fraction: float = _VAL_GAME_FRACTION,
    device: str | None = None,
    verbose: bool = True,
    loss_history: list | None = None,
) -> PolicyInference:
    """
    Train a PolicyNet on self-play records and return a PolicyInference wrapper.

    Training details
    ----------------
    - Train/val split by game_id (not by pick) to prevent intra-game leakage.
      The last val_game_fraction fraction of games (chronologically by game_id)
      forms the validation set.
    - Records with is_primary_team_pick=True (primary team's picks) and
      is_primary_team_pick=False (opponent's picks, stored from their perspective)
      are placed in separate mini-batches and interleaved within each epoch.
      This ensures the whose_turn flag and draft-position features receive clear
      gradient signal from each perspective separately.
    - KL divergence minimised via cross-entropy proxy.
    - Early stopping on validation cross-entropy with patience epochs.
    - L2 regularisation via Adam weight_decay.

    Parameters
    ----------
    records          : list of SelfPlayRecord (from ReplayBuffer.all_records())
    schema           : FeatureSchema from the trained FM (provides vocab + indices)
    lr               : Adam learning rate (default 1e-3)
    weight_decay     : L2 regularisation coefficient (default 1e-4)
    batch_size       : mini-batch size (default 512)
    max_epochs       : maximum training epochs (default 50)
    patience         : early stopping patience on val loss (default 5)
    val_game_fraction: fraction of games held out for validation (default 0.2)
    device           : torch device string; auto-selects cuda/mps/cpu if None
    verbose          : print epoch-level loss logs (default True)
    loss_history     : if provided (pass an empty list), (epoch, train_kl, val_kl)
                       tuples are appended at each epoch for external plotting.

    Returns
    -------
    PolicyInference ready for use as MCTS prior (on CPU).
    """
    if not records:
        raise ValueError("No records provided for training.")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    # Train / val split by game
    all_game_ids = sorted({r.game_id for r in records})
    n_val = max(1, int(len(all_game_ids) * val_game_fraction))
    val_ids = set(all_game_ids[-n_val:])
    train_recs = [r for r in records if r.game_id not in val_ids]
    val_recs = [r for r in records if r.game_id in val_ids]

    if not train_recs:
        raise ValueError(
            f"No training records after val split ({len(val_ids)} val games, "
            f"{len(all_game_ids)} total games)."
        )

    V = len(schema.vocab)
    n_input = schema.n_features + POLICY_EXTRA_FEATURES

    if verbose:
        print(
            f"train_policy: {len(train_recs)} train / {len(val_recs)} val records  "
            f"({len(all_game_ids) - n_val} / {n_val} games)  "
            f"n_input={n_input}  V={V}  device={device}"
        )

    train_X, train_mask, train_target = _build_arrays(train_recs, schema, V, n_input)
    val_X, val_mask, val_target = _build_arrays(val_recs, schema, V, n_input)

    net = PolicyNet(n_input, V)
    net.to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # Separate indices by primary vs opponent team picks (if field available)
    has_flag = hasattr(train_recs[0], "is_primary_team_pick")
    if has_flag:
        primary_idx = np.array(
            [i for i, r in enumerate(train_recs) if r.is_primary_team_pick], dtype=np.intp
        )
        opponent_idx = np.array(
            [i for i, r in enumerate(train_recs) if not r.is_primary_team_pick], dtype=np.intp
        )
    else:
        primary_idx = np.arange(len(train_recs), dtype=np.intp)
        opponent_idx = np.array([], dtype=np.intp)

    # ── Val loss helper ───────────────────────────────────────────────────────
    def _val_loss() -> float:
        net.eval()
        total, n_batches = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_recs), batch_size):
                end = min(start + batch_size, len(val_recs))
                x = torch.from_numpy(val_X[start:end]).to(dev)
                m = torch.from_numpy(val_mask[start:end]).to(dev)
                t = torch.from_numpy(val_target[start:end]).to(dev)
                total += _kl_loss(net, x, m, t).item()
                n_batches += 1
        net.train()
        return total / max(1, n_batches)

    best_val = float("inf")
    patience_count = 0
    best_weights: dict | None = None

    for epoch in range(max_epochs):
        net.train()
        rng_ep = np.random.default_rng(epoch)

        # Shuffle each group and interleave batches
        perm_p = rng_ep.permutation(primary_idx) if len(primary_idx) > 0 else np.array([], dtype=np.intp)
        perm_o = rng_ep.permutation(opponent_idx) if len(opponent_idx) > 0 else np.array([], dtype=np.intp)
        batches_p = _make_batches(perm_p, batch_size)
        batches_o = _make_batches(perm_o, batch_size)

        all_batches = []
        for j in range(max(len(batches_p), len(batches_o))):
            if j < len(batches_p):
                all_batches.append(batches_p[j])
            if j < len(batches_o):
                all_batches.append(batches_o[j])

        epoch_loss, n_b = 0.0, 0
        for batch_idx in all_batches:
            x = torch.from_numpy(train_X[batch_idx]).to(dev)
            m = torch.from_numpy(train_mask[batch_idx]).to(dev)
            t = torch.from_numpy(train_target[batch_idx]).to(dev)
            optimizer.zero_grad()
            loss = _kl_loss(net, x, m, t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        val = _val_loss()
        train_kl = epoch_loss / max(1, n_b)
        if loss_history is not None:
            loss_history.append((epoch, train_kl, val))
        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:3d}  train={train_kl:.4f}  val={val:.4f}")

        if val < best_val - 1e-6:
            best_val = val
            patience_count = 0
            best_weights = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  (best val={best_val:.4f})")
                break

    if best_weights is not None:
        net.load_state_dict(best_weights)

    net.eval()
    net.cpu()
    return PolicyInference(net=net, schema=schema)
