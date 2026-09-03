"""
joint_net.py — Part 5.2: Joint Policy+Value Network

Single MLP with a shared trunk + two output heads. Trained from scratch on the
full self-play replay buffer. Supersedes policy_best.pkl as the MCTS prior and
adds a value head that enables rollout-free MCTS.

Architecture
------------
  input (n_features+7) → Linear(512) → ReLU → Linear(256) → ReLU  [shared trunk]
    → policy head: Linear(128) → ReLU → Linear(V)                  [logits; masked softmax at inference]
    → value head:  Linear(128) → ReLU → Linear(1) → Sigmoid        [win probability ∈ (0,1)]

Rollout-free MCTS
-----------------
  Existing flow:  leaf → rollout (random fills + FM call at terminal) → backprop
  New flow:       leaf → value_net.evaluate(leaf.state) → backprop

  The value head replaces the entire rollout. It was trained on terminal_win_prob
  labels from self-play games, so it implicitly averages over many possible
  continuations. Lower variance → faster tree convergence.

  Pass the same JointNetInference as both `policy` and `value_net` to _run_mcts /
  recommend() to get both benefits: learned PUCT prior + rollout-free evaluation.

Caching
-------
  No LRU cache is applied to evaluate(). In MCTS each node is a unique partial
  state that is evaluated once (at first leaf visit), after which the node is
  expanded and Q values are tracked in the tree. The only cache hit would be a
  transposition (two paths reaching the same composition) — rare in ordered draft
  trees. The existing FMEvaluator cache for terminal states is bypassed in
  rollout-free mode, which is acceptable: rollout-free trees rarely reach
  terminal nodes because the tree evaluates every leaf directly.

Training
--------
  loss = policy_cross_entropy(logits, visit_dist) + lambda_v * bce(value_pred, win_prob)
  Augmented (team-swapped) records: contribute only to value head (policy weight=0).
  Original records: contribute to both heads.
  Same train/val split by game_id as train_policy. Early stopping on combined val loss.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bsdraft.features.engineering import FeatureSchema
from bsdraft.mcts.state import DraftState
from bsdraft.selfplay.policy_net import (
    POLICY_EXTRA_FEATURES,
    _make_batches,
    _picker_pov,
    encode_partial_state,
)

# augment_with_team_swap is imported lazily inside train_joint() to break the
# circular import chain: joint_net → self_play → joint_net.

# src/bsdraft/<subpackage>/<module>.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
JOINT_NET_PATH = REPO_ROOT / "data" / "joint_net.pkl"


# ── Terminal-state encoding helper ────────────────────────────────────────────

def _encode_terminal_for_value(state: DraftState, schema: FeatureSchema) -> np.ndarray:
    """
    Encode a terminal DraftState for value head inference.

    encode_partial_state() calls state.whose_turn, which raises ValueError
    at pick_number=6. This function replicates the same encoding but hard-codes
    pick_number=5 (last non-terminal depth) and whose_turn="mine", matching the
    deepest pick depth present in the training data.

    The value head was trained on records up to pick depth 5. Terminal states
    are compositions the FM evaluates as labels — the value head at pick depth 5
    is the closest training distribution and generalises well to 6-pick states.
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

    # whose_turn = "mine" (hard-coded; terminal state has no current picker)
    vec[schema.n_features] = 1.0
    # pick_number one-hot at position 5 (last training depth)
    vec[schema.n_features + 1 + 5] = 1.0

    return vec

# ── Hyperparameters ───────────────────────────────────────────────────────────

_TRUNK_HIDDEN: tuple[int, ...] = (512, 256)
_HEAD_HIDDEN: int = 128
_LR: float = 1e-3
_WEIGHT_DECAY: float = 1e-4
_BATCH_SIZE: int = 512
_MAX_EPOCHS: int = 50
_PATIENCE: int = 5
_VAL_GAME_FRACTION: float = 0.2
_LAMBDA_V: float = 1.0  # value head loss weight relative to policy head


# ── Model ─────────────────────────────────────────────────────────────────────

class JointNet(nn.Module):
    """
    Shared trunk MLP with separate policy and value heads.

    Forward pass returns (policy_logits, value) where:
      policy_logits : (B, V) — raw logits over all vocabulary brawlers
      value         : (B, 1) — win probability after sigmoid, ∈ (0, 1)

    Masking to available brawlers is done externally at inference time so the
    same model weights can be reused with different availability sets.
    """

    def __init__(
        self,
        n_input: int,
        n_vocab: int,
        trunk_hidden: tuple[int, ...] = _TRUNK_HIDDEN,
        head_hidden: int = _HEAD_HIDDEN,
    ) -> None:
        super().__init__()

        # Shared trunk: fully connected layers up to the last trunk width.
        trunk_layers: list[nn.Module] = []
        prev = n_input
        for h in trunk_hidden:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = prev

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(trunk_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_vocab),
        )

        # Value head — outputs a single win-probability after sigmoid
        self.value_head = nn.Sequential(
            nn.Linear(trunk_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, n_input) → (policy_logits (B, V), value (B, 1))"""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)


# ── Inference wrapper ─────────────────────────────────────────────────────────

@dataclass
class JointNetInference:
    """
    Inference wrapper that implements both PolicyInference and FMEvaluator interfaces:
      .predict_prior(state, available) → np.ndarray  — PUCT prior for MCTS expansion
      .evaluate(state)                  → float       — win probability at any leaf state

    This object can be passed as both `policy` and `value_net` to _run_mcts /
    recommend(), enabling rollout-free MCTS with a learned prior.

    predict_prior mirrors the state to the picker's perspective before encoding
    (matching how self-play records are stored, all with whose_turn=="mine").

    evaluate does NOT mirror — it encodes state as-is. The value head was trained
    on both original and team-swapped records and correctly returns P(my_team wins)
    regardless of whose_turn.
    """

    net: JointNet
    schema: FeatureSchema

    def predict_prior(
        self,
        state: DraftState,
        available: list[str],
    ) -> np.ndarray:
        """
        Return a normalized probability vector over `available` brawlers.
        Mirrors state to picker's perspective if whose_turn == 'opp'.
        Drop-in replacement for PolicyInference.predict_prior.
        """
        if not available:
            return np.array([], dtype=np.float32)

        eval_state = _picker_pov(state)
        vec = encode_partial_state(eval_state, self.schema)
        x = torch.from_numpy(vec).unsqueeze(0)

        vocab_idx = self.schema._vocab_idx
        V = len(self.schema.vocab)

        avail_indices = [vocab_idx[b] for b in available if b in vocab_idx]
        if not avail_indices:
            return np.ones(len(available), dtype=np.float32) / len(available)

        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits[0]  # (V,)

        mask = torch.full((V,), float("-inf"))
        mask[avail_indices] = 0.0
        probs = torch.softmax(logits + mask, dim=0).numpy()

        result = np.array(
            [probs[vocab_idx[b]] if b in vocab_idx else 0.0 for b in available],
            dtype=np.float32,
        )
        s = result.sum()
        return result / s if s > 0 else np.ones(len(available), dtype=np.float32) / len(available)

    def evaluate(self, state: DraftState) -> float:
        """
        Return P(my_team wins) ∈ (0, 1) for any partial or terminal DraftState.

        Encodes state as-is (no mirroring). The value head was trained on both
        original and team-swapped records and returns P(my_team wins) for any
        orientation.

        Terminal states are handled by substituting whose_turn="mine" (the
        value head was trained at pick depth 5 so depth-6 is extrapolated).

        Drop-in replacement for FMEvaluator.evaluate (works on partial states too).
        """
        # encode_partial_state accesses state.whose_turn, which raises for
        # terminal states (pick_number == 6). Use the dedicated terminal encoder.
        if state.is_terminal:
            vec = _encode_terminal_for_value(state, self.schema)
        else:
            vec = encode_partial_state(state, self.schema)
        x = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            _, value = self.net(x)
        return float(value[0, 0])

    def save(self, path: Path = JOINT_NET_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"JointNetInference saved → {path}")

    @classmethod
    def load(cls, path: Path = JOINT_NET_PATH) -> JointNetInference:
        import importlib

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "__main__":
                    for mod_name in ("joint_net", "src.joint_net"):
                        try:
                            return getattr(importlib.import_module(mod_name), name)
                        except (ModuleNotFoundError, AttributeError):
                            pass
                return super().find_class(module, name)

        with open(path, "rb") as fh:
            return _Unpickler(fh).load()


# ── Training helpers ──────────────────────────────────────────────────────────

def _build_arrays_joint(
    records: list,
    schema: FeatureSchema,
    V: int,
    n_input: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training arrays for joint policy+value training.

    Returns
    -------
    X            : (N, n_input) float32 — encoded partial states
    avail_mask   : (N, V) bool — True where brawler is available
    policy_targets : (N, V) float32 — visit distribution (0 for augmented records)
    value_targets  : (N,) float32 — terminal win probabilities
    policy_weights : (N,) float32 — 1.0 for original records, 0.0 for augmented
                     (augmented records have empty visit_dist and should not
                      contribute to policy head loss)
    """
    N = len(records)
    X = np.zeros((N, n_input), dtype=np.float32)
    avail_mask = np.zeros((N, V), dtype=bool)
    policy_targets = np.zeros((N, V), dtype=np.float32)
    value_targets = np.zeros(N, dtype=np.float32)
    policy_weights = np.zeros(N, dtype=np.float32)
    vocab_idx = schema._vocab_idx
    vocab = schema.vocab

    for i, rec in enumerate(records):
        X[i] = encode_partial_state(rec.state, schema)
        value_targets[i] = rec.terminal_win_prob

        used = rec.state.my_team | rec.state.opp_team | rec.state.bans
        for j, b in enumerate(vocab):
            if b not in used:
                avail_mask[i, j] = True

        # Augmented records have empty visit_dist — skip policy supervision.
        if rec.visit_dist:
            policy_weights[i] = 1.0
            visit_mass = sum(rec.visit_dist.values())
            if visit_mass > 0:
                for b, frac in rec.visit_dist.items():
                    idx = vocab_idx.get(b)
                    if idx is not None and avail_mask[i, idx]:
                        policy_targets[i, idx] = frac / visit_mass
            # Fallback: if no valid picks found (shouldn't happen), uniform.
            if policy_targets[i].sum() == 0:
                n_avail = int(avail_mask[i].sum())
                if n_avail > 0:
                    policy_targets[i, avail_mask[i]] = 1.0 / n_avail

    return X, avail_mask, policy_targets, value_targets, policy_weights


def _joint_loss(
    net: JointNet,
    x: torch.Tensor,
    avail_mask: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    policy_weights: torch.Tensor,
    lambda_v: float,
) -> tuple[torch.Tensor, float, float]:
    """
    Combined policy cross-entropy + value BCE loss.

    Policy loss is zero-weighted for augmented records (policy_weights=0).
    Value loss is computed for all records (original + augmented).

    Returns
    -------
    (combined_loss, policy_component_float, value_component_float)
      combined_loss           : torch.Tensor — use for .backward()
      policy_component_float  : policy_loss scalar (for logging/plotting)
      value_component_float   : lambda_v * value_loss scalar (for logging/plotting)
    """
    policy_logits, value_pred = net(x)

    # ── Policy head: weighted cross-entropy vs visit distribution ─────────────
    masked_logits = policy_logits.masked_fill(~avail_mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    safe_log_probs = log_probs.masked_fill(policy_targets == 0, 0.0)
    per_sample_policy = -(policy_targets * safe_log_probs).sum(dim=-1)  # (B,)
    policy_loss = (policy_weights * per_sample_policy).sum() / (policy_weights.sum() + 1e-8)

    # ── Value head: binary cross-entropy vs terminal win probability ──────────
    value_loss = F.binary_cross_entropy(value_pred.squeeze(-1), value_targets)

    combined = policy_loss + lambda_v * value_loss
    return combined, float(policy_loss.detach()), float(lambda_v * value_loss.detach())


# ── Training ──────────────────────────────────────────────────────────────────

def train_joint(
    records: list,
    schema: FeatureSchema,
    *,
    lambda_v: float = _LAMBDA_V,
    lr: float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
    batch_size: int = _BATCH_SIZE,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
    val_game_fraction: float = _VAL_GAME_FRACTION,
    trunk_hidden: tuple[int, ...] | None = None,
    device: str | None = None,
    verbose: bool = True,
    loss_history: list | None = None,
) -> JointNetInference:
    """
    Train a JointNet on self-play records and return a JointNetInference wrapper.

    Augmented (team-swapped) records are generated internally and contribute
    only to the value head. Original records contribute to both heads.

    Parameters
    ----------
    records          : original self-play records from ReplayBuffer.all_records()
    schema           : FeatureSchema from the trained FM
    lambda_v         : value head loss weight (default 1.0 = equal weight)
    lr               : Adam learning rate
    weight_decay     : L2 regularisation
    batch_size       : mini-batch size
    max_epochs       : maximum epochs
    patience         : early stopping patience on combined val loss
    val_game_fraction: fraction of games held out for validation
    device           : torch device; auto-selects cuda/mps/cpu if None
    verbose          : print epoch-level logs
    loss_history     : if provided (pass empty list), appends per-epoch tuples:
                       (epoch, train_policy, train_value, val_combined)
                       where train_policy = policy cross-entropy contribution,
                       train_value = lambda_v * value BCE contribution.
                       These two separate curves are useful for diagnosing which
                       head is driving training and for the notebook calibration plots.

    Returns
    -------
    JointNetInference on CPU, ready for MCTS use.
    """
    if not records:
        raise ValueError("No records provided for training.")

    # Lazy import to avoid circular dependency (self_play imports joint_net).
    try:
        from bsdraft.selfplay.generate import augment_with_team_swap
    except ModuleNotFoundError:
        from bsdraft.selfplay.generate import augment_with_team_swap

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    # ── Train / val split by game_id ──────────────────────────────────────────
    all_game_ids = sorted({r.game_id for r in records})
    n_val = max(1, int(len(all_game_ids) * val_game_fraction))
    val_ids = set(all_game_ids[-n_val:])
    train_orig = [r for r in records if r.game_id not in val_ids]
    val_orig   = [r for r in records if r.game_id in val_ids]

    if not train_orig:
        raise ValueError("No training records after val split.")

    # Augment: team-swapped copies for value head only.
    # Augmented records have empty visit_dist → policy_weights=0 in arrays.
    train_aug = augment_with_team_swap(train_orig)
    val_aug   = augment_with_team_swap(val_orig)

    train_all = train_orig + train_aug
    val_all   = val_orig   + val_aug

    V = len(schema.vocab)
    n_input = schema.n_features + POLICY_EXTRA_FEATURES

    if verbose:
        print(
            f"train_joint: {len(train_orig)} original + {len(train_aug)} augmented train  "
            f"/ {len(val_orig)} + {len(val_aug)} augmented val  "
            f"({len(all_game_ids) - n_val} / {n_val} games)  "
            f"n_input={n_input}  V={V}  device={device}  lambda_v={lambda_v}"
        )

    train_X, train_mask, train_pol, train_val, train_pw = _build_arrays_joint(
        train_all, schema, V, n_input
    )
    val_X, val_mask, val_pol, val_val, val_pw = _build_arrays_joint(
        val_all, schema, V, n_input
    )

    net = JointNet(n_input, V, trunk_hidden=trunk_hidden if trunk_hidden is not None else _TRUNK_HIDDEN)
    net.to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Val loss helper ───────────────────────────────────────────────────────
    def _val_loss() -> float:
        net.eval()
        total, n_b = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_all), batch_size):
                end = min(start + batch_size, len(val_all))
                combined, _, _ = _joint_loss(
                    net,
                    torch.from_numpy(val_X[start:end]).to(dev),
                    torch.from_numpy(val_mask[start:end]).to(dev),
                    torch.from_numpy(val_pol[start:end]).to(dev),
                    torch.from_numpy(val_val[start:end]).to(dev),
                    torch.from_numpy(val_pw[start:end]).to(dev),
                    lambda_v,
                )
                total += combined.item()
                n_b += 1
        net.train()
        return total / max(1, n_b)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    patience_count = 0
    best_weights: dict | None = None

    idx_all = np.arange(len(train_all), dtype=np.intp)

    for epoch in range(max_epochs):
        net.train()
        perm = np.random.default_rng(epoch).permutation(idx_all)
        batches = _make_batches(perm, batch_size)

        epoch_pol, epoch_val_head, n_b = 0.0, 0.0, 0
        for batch_idx in batches:
            loss, pol_item, val_item = _joint_loss(
                net,
                torch.from_numpy(train_X[batch_idx]).to(dev),
                torch.from_numpy(train_mask[batch_idx]).to(dev),
                torch.from_numpy(train_pol[batch_idx]).to(dev),
                torch.from_numpy(train_val[batch_idx]).to(dev),
                torch.from_numpy(train_pw[batch_idx]).to(dev),
                lambda_v,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_pol += pol_item
            epoch_val_head += val_item
            n_b += 1

        val = _val_loss()
        train_pol_avg = epoch_pol / max(1, n_b)
        train_val_avg = epoch_val_head / max(1, n_b)
        if loss_history is not None:
            loss_history.append((epoch, train_pol_avg, train_val_avg, val))
        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:3d}  policy={train_pol_avg:.4f}  value={train_val_avg:.4f}  val={val:.4f}")

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
    return JointNetInference(net=net, schema=schema)
