"""
CoverageBelief — the monotone per-slot coverage filter, design doc Section 1.3.

b_t = (kappa_t, {p_j,t}_{j<=K_t}, {(alpha_a, beta_a)}_{a in action_modes})

p_j,t = P(c_j,t = 1 | o_1:t) is updated by the monotone recursion
    p_j,t = 1 - (1 - p_j,t-1) * (1 - g_{a_t}(m_j,t))
(coverage can only increase: docs are never dropped). Unidentified slots
j > K_t have p_j,t := 0 by definition (requirements not yet found are
uncovered), which is what lets early bridge discovery raise C_t later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .action_labeler import ActionLabel, ActionMode
from .config import ACECConfig
from .k_posterior import KPosterior, KPredictor, UniformKPredictor
from .observation_model import HitRateBeta, ObservationModel
from .slots import SlotStore


@dataclass
class CoverageFeatures:
    """features <- [C, sqrt(Var(C)), E[K], H(kappa), p_1..p_Kmax (padded), bound flags, t]
    (Algorithm 1, line 12) — the policy-input / logging view of the belief."""

    coverage: float
    coverage_std: float
    expected_k: float
    k_entropy: float
    slot_p: List[float]
    slot_bound: List[bool]
    turn: int

    def to_vector(self, k_max: int) -> List[float]:
        padded_p = (self.slot_p + [0.0] * k_max)[:k_max]
        padded_bound = ([1.0 if b else 0.0 for b in self.slot_bound] + [0.0] * k_max)[:k_max]
        return [
            self.coverage,
            self.coverage_std,
            self.expected_k,
            self.k_entropy,
            *padded_p,
            *padded_bound,
            float(self.turn),
        ]


class CoverageBelief:
    """Owns the slot store, the K posterior, the per-slot coverage posteriors,
    and the action-indexed hit-rate Betas for a single question's episode."""

    def __init__(
        self,
        config: Optional[ACECConfig] = None,
        k_predictor: Optional[KPredictor] = None,
        obs_model: Optional[ObservationModel] = None,
    ):
        self.config = config or ACECConfig()
        self.slots = SlotStore()
        self.p: List[float] = []
        self.k_predictor = k_predictor or UniformKPredictor(self.config.k_max)
        self.obs_model = obs_model or ObservationModel()
        self.hit_betas: Dict[str, HitRateBeta] = {
            mode: HitRateBeta(
                self.config.hit_prior_alpha0.get(mode, 1.0),
                self.config.hit_prior_beta0.get(mode, 1.0),
                ess_cap=self.config.ess_cap,
            )
            for mode in self.config.action_modes
        }
        self.turn = 0
        self.kappa: Optional[KPosterior] = None

    def reset(self, question: str = "") -> None:
        self.slots = SlotStore()
        self.p = []
        self.turn = 0
        prior = self.k_predictor.predict(question)
        self.kappa = KPosterior(self.config.k_max, prior)
        for beta in self.hit_betas.values():
            beta.reset()

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------
    def spawn_slot(self, hypothesis: str) -> int:
        slot = self.slots.spawn(hypothesis, self.turn)
        self.p.append(0.0)
        if self.kappa is not None:
            self.kappa.truncate_ge(len(self.slots))
        return slot.idx

    def bind_slot(self, idx: int, entity: str) -> None:
        self.slots[idx].bind(entity, self.turn)

    # ------------------------------------------------------------------
    # Belief update tau(b_t, a_t, o_t)
    # ------------------------------------------------------------------
    def step(self, action: ActionLabel, slot_scores: Dict[int, float]) -> CoverageFeatures:
        """
        One belief update (Algorithm 1, lines 4-11): given the routed action
        and this turn's per-slot NLI entailment scores m_j, apply the monotone
        coverage recursion and, for the targeted slot only, update that
        action's hit-rate Beta.
        """
        self.turn += 1
        mode = action.mode.value if isinstance(action.mode, ActionMode) else str(action.mode)
        beta = self.hit_betas.get(mode)
        pi_a = beta.mean if beta is not None else 0.5

        for j, m in slot_scores.items():
            if j >= len(self.p):
                continue
            role = "tgt" if j == action.target_slot else "inc"
            bound = self.slots[j].bound
            g = self.obs_model.hit_posterior(m, pi_a, role=role, bound=bound)
            self.p[j] = 1.0 - (1.0 - self.p[j]) * (1.0 - g)
            if role == "tgt" and beta is not None:
                beta.update(g)

        return self.features()

    # ------------------------------------------------------------------
    # Coverage functional and uncertainty (Section 1.3)
    # ------------------------------------------------------------------
    def coverage(self) -> float:
        """C_t = E[fraction covered] = sum_k kappa(k) * (1/k) * sum_{j<=k} p_j."""
        if self.kappa is None or not self.p:
            return 0.0
        total = 0.0
        for k in range(1, self.config.k_max + 1):
            prob_k = self.kappa.kappa[k - 1]
            if prob_k <= 0:
                continue
            covered = sum(self._p_at(j) for j in range(k)) / k
            total += prob_k * covered
        return float(total)

    def coverage_variance(self) -> float:
        """Var[C_t] = within-K (slot Bernoulli variance) + across-K (variance
        of the per-K coverage mean under kappa) — Section 1.3."""
        if self.kappa is None or not self.p:
            return 0.0
        k_max = self.config.k_max
        within = 0.0
        per_k_mean = []
        for k in range(1, k_max + 1):
            prob_k = self.kappa.kappa[k - 1]
            slot_p = [self._p_at(j) for j in range(k)]
            slot_var = sum(pj * (1.0 - pj) for pj in slot_p)
            within += prob_k * slot_var / (k * k)
            per_k_mean.append(sum(slot_p) / k)
        mean_of_means = sum(self.kappa.kappa[k - 1] * per_k_mean[k - 1] for k in range(1, k_max + 1))
        across = sum(
            self.kappa.kappa[k - 1] * (per_k_mean[k - 1] - mean_of_means) ** 2 for k in range(1, k_max + 1)
        )
        return float(within + across)

    def _p_at(self, j: int) -> float:
        # Unidentified slots (j >= K_t) are uncovered by definition (Section 1.3).
        return self.p[j] if j < len(self.p) else 0.0

    def features(self) -> CoverageFeatures:
        return CoverageFeatures(
            coverage=self.coverage(),
            coverage_std=math.sqrt(max(self.coverage_variance(), 0.0)),
            expected_k=self.kappa.expectation() if self.kappa else 0.0,
            k_entropy=self.kappa.entropy() if self.kappa else 0.0,
            slot_p=list(self.p),
            slot_bound=[s.bound for s in self.slots],
            turn=self.turn,
        )

    # ------------------------------------------------------------------
    # Downstream signals
    # ------------------------------------------------------------------
    def suggest_target_slot(self) -> Optional[int]:
        """Lowest-p_j unbound slot — the mechanism aimed at the design doc's
        +27.9pt bridge-injection headroom (Section 2.1), now posterior-driven
        instead of oracle-driven."""
        candidates = [(self._p_at(j), j) for j, s in enumerate(self.slots) if not s.bound]
        if not candidates:
            return None
        return min(candidates)[1]

    def should_stop_voi(self) -> bool:
        """
        Interpretable value-of-information stopping diagnostic (Section 2.1):
            STOP if eta * max_{(m,j)} omega_j * (1 - p_j) * pi_hat_{(m,j)} < c_r
        Exact under the generative model because E_m[g_a(m)] = pi_hat_a (a
        posterior's prior-predictive mean equals the prior). Myopic — it
        undervalues DECOMPOSE — so this is a diagnostic/gate, not the trained
        policy.
        """
        if self.kappa is None or not self.p:
            return False
        best_value = 0.0
        for j in range(len(self.p)):
            omega_j = self.kappa.weight_slot_le_k(j + 1)
            best_mode_pi = max((b.mean for b in self.hit_betas.values()), default=0.0)
            value = omega_j * (1.0 - self._p_at(j)) * best_mode_pi
            best_value = max(best_value, value)
        return (self.config.eta * best_value) < self.config.c_r
