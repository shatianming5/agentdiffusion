"""Rich Order Decoder: convert 200-dim agent states to structured order predictions.

Given two consecutive agent states (t and t+1), predict what orders each agent
would place during the transition.  Uses a DETR-style cross-attention decoder
with learnable order query slots and per-query structured output heads.

Architecture:
    1. State projection:  concat(state_t, state_t1, delta) -> 600-dim -> d_model
    2. Order query slots:  N_queries learnable queries (like DETR object queries)
    3. Cross-attention:    queries attend to projected state
    4. Per-query heads:    direction / price / size / type / urgency / active
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Output field layout (total = 10 per query)
# ---------------------------------------------------------------------------
#   [0:3]  direction  3-class logits  (buy / sell / hold)
#   [3]    price      relative price offset (regression)
#   [4]    size       log order size        (regression)
#   [5:8]  type       3-class logits  (limit / market / cancel)
#   [8]    urgency    0-1 urgency score     (regression)
#   [9]    active     binary logit          (is slot active?)
# ---------------------------------------------------------------------------
_DIR_SLICE = slice(0, 3)
_PRICE_IDX = 3
_SIZE_IDX = 4
_TYPE_SLICE = slice(5, 8)
_URGENCY_IDX = 8
_ACTIVE_IDX = 9
_D_ORDER_OUT = 10


class RichOrderDecoder(nn.Module):
    """Decode a 200-dim agent state into concrete order predictions.

    Given two consecutive agent states (t and t+1), predict what orders
    this agent would place during the transition.

    Architecture:
        1. State projection: concat(state_t, state_t1, delta) -> 600-dim -> d_model
        2. Order query slots: N_queries=32 learnable queries (like DETR)
        3. Cross-attention: queries attend to projected state
        4. Per-query output heads:
           - direction_head: 3-class (buy/sell/hold) -> logits [3]
           - price_head: regression -> relative price offset [1]
           - size_head: regression -> log order size [1]
           - type_head: 3-class (limit/market/cancel) -> logits [3]
           - urgency_head: regression -> 0-1 urgency score [1]
           - active_head: binary -> is this query slot active? [1]

        Total per-query output: 10 dims

    Input:
        state_t:  [B, d_state=200]
        state_t1: [B, d_state=200]
    Output:
        orders: [B, N_queries, 10]

    For sequence decoding:
        states: [B, T, H, W, d_state=200]
        -> orders: [B, T-1, H*W, N_queries, 10]  (per-cell, per-query)
    """

    def __init__(
        self,
        d_state: int = 200,
        d_model: int = 256,
        n_queries: int = 32,
        n_layers: int = 3,
        n_heads: int = 8,
        d_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.n_queries = n_queries

        # --- State projection (3 * d_state -> d_model) ---
        self.state_proj = nn.Sequential(
            nn.Linear(d_state * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # --- Learnable order query slots ---
        self.order_queries = nn.Parameter(
            torch.randn(n_queries, d_model) * 0.02
        )

        # --- Transformer decoder (queries cross-attend to state memory) ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # --- Per-query output heads ---
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 3),
        )
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.size_head = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 3),
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.active_head = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Bias the active head towards "inactive" so most slots start quiet
        nn.init.constant_(self.active_head[-1].bias, -2.0)

    # ------------------------------------------------------------------
    def _assemble_output(self, decoded: torch.Tensor) -> torch.Tensor:
        """Apply per-query heads and concatenate into [B, Q, 10]."""
        direction = self.direction_head(decoded)          # [B, Q, 3]
        price = self.price_head(decoded)                  # [B, Q, 1]
        size = self.size_head(decoded)                    # [B, Q, 1]
        order_type = self.type_head(decoded)              # [B, Q, 3]
        urgency = self.urgency_head(decoded)              # [B, Q, 1]
        active = self.active_head(decoded)                # [B, Q, 1]

        return torch.cat(
            [direction, price, size, order_type, urgency, active], dim=-1
        )  # [B, Q, 10]

    # ------------------------------------------------------------------
    def forward(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict orders from a single agent state transition.

        Args:
            state_t:  [B, d_state]  (or [B, N_cells, d_state] for batched cells)
            state_t1: [B, d_state]  (same shape)

        Returns:
            orders: [B, N_queries, 10]  (or [B, N_cells, N_queries, 10])
        """
        # Handle both flat [B, d_state] and spatial [B, N, d_state] inputs
        squeezed = False
        if state_t.dim() == 2:
            state_t = state_t.unsqueeze(1)    # [B, 1, d_state]
            state_t1 = state_t1.unsqueeze(1)
            squeezed = True

        B, N, D = state_t.shape
        delta = state_t1 - state_t
        x = torch.cat([state_t, state_t1, delta], dim=-1)  # [B, N, 3*d_state]

        memory = self.state_proj(x)  # [B, N, d_model]

        # Expand queries for batch
        queries = self.order_queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: queries attend to memory
        decoded = self.decoder(queries, memory)  # [B, n_queries, d_model]

        out = self._assemble_output(decoded)  # [B, n_queries, 10]

        if squeezed:
            return out  # [B, n_queries, 10]
        # When N>1 we return separate query sets per memory token:
        # this path is used by decode_sequence below.
        return out

    # ------------------------------------------------------------------
    def forward_cells(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Decode per-cell order predictions for a spatial grid.

        Args:
            state_t:  [B, H, W, d_state]
            state_t1: [B, H, W, d_state]

        Returns:
            orders: [B, H*W, N_queries, 10]
        """
        B, H, W, D = state_t.shape
        N = H * W

        # Flatten cells
        st = state_t.reshape(B * N, D)
        st1 = state_t1.reshape(B * N, D)

        # Forward each cell independently
        out = self.forward(st, st1)  # [B*N, n_queries, 10]
        return out.reshape(B, N, self.n_queries, _D_ORDER_OUT)

    # ------------------------------------------------------------------
    def decode_sequence(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a full temporal sequence, per cell, per query.

        Args:
            states: [B, T, H, W, d_state]

        Returns:
            orders: [B, T-1, H*W, N_queries, 10]
        """
        B, T, H, W, D = states.shape
        N = H * W

        s_t = states[:, :-1]   # [B, T-1, H, W, D]
        s_t1 = states[:, 1:]   # [B, T-1, H, W, D]

        # Merge batch and time
        s_t = s_t.reshape(B * (T - 1), H, W, D)
        s_t1 = s_t1.reshape(B * (T - 1), H, W, D)

        out = self.forward_cells(s_t, s_t1)  # [B*(T-1), H*W, n_queries, 10]
        return out.reshape(B, T - 1, N, self.n_queries, _D_ORDER_OUT)


# ======================================================================
# Matching utilities
# ======================================================================

@torch.no_grad()
def _compute_cost_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_cls: float = 1.0,
    w_reg: float = 1.0,
) -> torch.Tensor:
    """Build Q x M cost matrix for Hungarian matching.

    Args:
        pred:   [Q, 10]   predicted order fields (one sample, one cell).
        target: [M, 10]   ground-truth orders for this cell/window.

    Returns:
        cost: [Q, M]  (lower = better match).
    """
    Q = pred.shape[0]
    M = target.shape[0]

    # Direction cost: negative log-prob under softmax
    dir_logp = F.log_softmax(pred[:, _DIR_SLICE], dim=-1)       # [Q, 3]
    dir_gt = target[:, _DIR_SLICE].argmax(dim=-1)               # [M]
    # Gather: for each (q, m), pick the log-prob of gt class
    cost_dir = -dir_logp[:, None, :].expand(Q, M, 3).gather(
        2, dir_gt[None, :, None].expand(Q, M, 1)
    ).squeeze(-1)  # [Q, M]

    # Type cost (same idea)
    type_logp = F.log_softmax(pred[:, _TYPE_SLICE], dim=-1)     # [Q, 3]
    type_gt = target[:, _TYPE_SLICE].argmax(dim=-1)             # [M]
    cost_type = -type_logp[:, None, :].expand(Q, M, 3).gather(
        2, type_gt[None, :, None].expand(Q, M, 1)
    ).squeeze(-1)  # [Q, M]

    # Price cost: L1
    cost_price = (
        pred[:, _PRICE_IDX].unsqueeze(1) - target[:, _PRICE_IDX].unsqueeze(0)
    ).abs()  # [Q, M]

    # Size cost: L1
    cost_size = (
        pred[:, _SIZE_IDX].unsqueeze(1) - target[:, _SIZE_IDX].unsqueeze(0)
    ).abs()  # [Q, M]

    cost = w_cls * (cost_dir + cost_type) + w_reg * (cost_price + cost_size)
    return cost


def _hungarian_match(cost: torch.Tensor) -> tuple[list[int], list[int]]:
    """Optimal assignment via scipy, falling back to greedy if unavailable."""
    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        return row_ind.tolist(), col_ind.tolist()
    return _greedy_match(cost)


def _greedy_match(cost: torch.Tensor) -> tuple[list[int], list[int]]:
    """Greedy matching fallback: pick lowest-cost pair iteratively."""
    cost_np = cost.clone()
    Q, M = cost_np.shape
    row_inds: list[int] = []
    col_inds: list[int] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    n_matches = min(Q, M)
    for _ in range(n_matches):
        # Mask out already-used rows and cols
        mask = cost_np.new_full((Q, M), float("inf"))
        for r in range(Q):
            for c in range(M):
                if r not in used_rows and c not in used_cols:
                    mask[r, c] = 0.0
        masked = cost_np + mask
        flat_idx = masked.reshape(-1).argmin().item()
        r, c = divmod(flat_idx, M)
        row_inds.append(r)
        col_inds.append(c)
        used_rows.add(r)
        used_cols.add(c)

    return row_inds, col_inds


# ======================================================================
# Loss
# ======================================================================

class RichOrderLoss(nn.Module):
    """Loss for training the Rich Order Decoder against real L3 order data.

    For each cell/agent, we have ground truth orders in that time window.
    We use Hungarian matching (like DETR) to match predicted query slots
    to GT orders, then compute:

    - direction:  cross-entropy
    - price:      smooth L1
    - size:       smooth L1
    - type:       cross-entropy
    - urgency:    smooth L1
    - active:     binary cross-entropy  (predicted active vs GT has-order)

    GT order tensor layout follows the same 10-dim convention:
        [0:3]  direction one-hot (or soft label)
        [3]    relative price offset
        [4]    log order size
        [5:8]  type one-hot
        [8]    urgency (0-1)
        [9]    1.0 (present flag, always 1 for real GT orders)

    Args:
        w_direction: weight for direction CE loss.
        w_price:     weight for price smooth-L1.
        w_size:      weight for size smooth-L1.
        w_type:      weight for type CE loss.
        w_urgency:   weight for urgency smooth-L1.
        w_active:    weight for active BCE loss.
        w_match_cls: weight for classification cost in Hungarian matching.
        w_match_reg: weight for regression cost in Hungarian matching.
    """

    def __init__(
        self,
        w_direction: float = 1.0,
        w_price: float = 1.0,
        w_size: float = 1.0,
        w_type: float = 1.0,
        w_urgency: float = 0.5,
        w_active: float = 2.0,
        w_match_cls: float = 1.0,
        w_match_reg: float = 1.0,
    ):
        super().__init__()
        self.w_direction = w_direction
        self.w_price = w_price
        self.w_size = w_size
        self.w_type = w_type
        self.w_urgency = w_urgency
        self.w_active = w_active
        self.w_match_cls = w_match_cls
        self.w_match_reg = w_match_reg

    # ------------------------------------------------------------------
    def _matched_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for a single (cell, timestep) pair.

        Args:
            pred:   [Q, 10]  decoder output for this cell.
            target: [M, 10]  GT orders for this cell (M can be 0).

        Returns:
            dict of scalar losses (on the same device as pred).
        """
        Q = pred.shape[0]
        M = target.shape[0]
        device = pred.device

        # --- Active loss: all queries get supervised ---
        # Build active target: matched queries -> 1, unmatched -> 0
        active_target = pred.new_zeros(Q)

        zero = pred.new_tensor(0.0)

        if M == 0:
            # No GT orders: all queries should be inactive
            loss_active = F.binary_cross_entropy_with_logits(
                pred[:, _ACTIVE_IDX], active_target
            )
            return {
                "direction": zero,
                "price": zero,
                "size": zero,
                "type": zero,
                "urgency": zero,
                "active": loss_active,
            }

        # --- Hungarian matching ---
        cost = _compute_cost_matrix(
            pred.detach(), target,
            w_cls=self.w_match_cls, w_reg=self.w_match_reg,
        )
        row_idx, col_idx = _hungarian_match(cost)

        # Active target for matched slots
        for r in row_idx:
            active_target[r] = 1.0

        loss_active = F.binary_cross_entropy_with_logits(
            pred[:, _ACTIVE_IDX], active_target
        )

        # --- Matched-pair losses (only on matched slots) ---
        if len(row_idx) == 0:
            return {
                "direction": zero,
                "price": zero,
                "size": zero,
                "type": zero,
                "urgency": zero,
                "active": loss_active,
            }

        row_t = torch.tensor(row_idx, device=device, dtype=torch.long)
        col_t = torch.tensor(col_idx, device=device, dtype=torch.long)

        matched_pred = pred[row_t]      # [K, 10]
        matched_gt = target[col_t]      # [K, 10]

        # Direction: CE on argmax of GT one-hot
        gt_dir = matched_gt[:, _DIR_SLICE].argmax(dim=-1)
        loss_dir = F.cross_entropy(matched_pred[:, _DIR_SLICE], gt_dir)

        # Price: smooth L1
        loss_price = F.smooth_l1_loss(
            matched_pred[:, _PRICE_IDX], matched_gt[:, _PRICE_IDX]
        )

        # Size: smooth L1
        loss_size = F.smooth_l1_loss(
            matched_pred[:, _SIZE_IDX], matched_gt[:, _SIZE_IDX]
        )

        # Type: CE on argmax of GT one-hot
        gt_type = matched_gt[:, _TYPE_SLICE].argmax(dim=-1)
        loss_type = F.cross_entropy(matched_pred[:, _TYPE_SLICE], gt_type)

        # Urgency: smooth L1 on sigmoid output vs GT (0-1)
        loss_urgency = F.smooth_l1_loss(
            matched_pred[:, _URGENCY_IDX].sigmoid(),
            matched_gt[:, _URGENCY_IDX],
        )

        return {
            "direction": loss_dir,
            "price": loss_price,
            "size": loss_size,
            "type": loss_type,
            "urgency": loss_urgency,
            "active": loss_active,
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target_orders: list[torch.Tensor],
        n_gt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute Rich Order Loss over a batch.

        Supports two calling conventions:

        (A) Padded tensor + count:
            pred:          [B, Q, 10]
            target_orders: single tensor [B, M_max, 10]  (zero-padded)
            n_gt:          [B]  int tensor, number of valid GT orders per sample

        (B) List of variable-length tensors:
            pred:          [B, Q, 10]
            target_orders: list of B tensors, each [M_i, 10]
            n_gt:          None

        Returns:
            dict with 'total', 'direction', 'price', 'size', 'type',
            'urgency', 'active' scalar losses averaged over the batch.
        """
        B = pred.shape[0]
        device = pred.device

        # Normalise target to list-of-tensors
        if isinstance(target_orders, torch.Tensor):
            # Convention (A): padded tensor
            assert n_gt is not None, (
                "n_gt required when target_orders is a padded tensor"
            )
            targets: list[torch.Tensor] = []
            for i in range(B):
                mi = int(n_gt[i].item())
                targets.append(target_orders[i, :mi])
        else:
            targets = target_orders

        # Accumulate per-sample losses
        keys = ["direction", "price", "size", "type", "urgency", "active"]
        accum = {k: pred.new_tensor(0.0) for k in keys}

        for i in range(B):
            sample_loss = self._matched_loss(pred[i], targets[i].to(device))
            for k in keys:
                accum[k] = accum[k] + sample_loss[k]

        # Average over batch
        for k in keys:
            accum[k] = accum[k] / max(B, 1)

        # Weighted total
        accum["total"] = (
            self.w_direction * accum["direction"]
            + self.w_price * accum["price"]
            + self.w_size * accum["size"]
            + self.w_type * accum["type"]
            + self.w_urgency * accum["urgency"]
            + self.w_active * accum["active"]
        )

        return accum
