"""
BeliefState: Bayesian belief over hidden environment parameters θ.

θ components (all approximated as Beta distributions over success probability):
  - θ_ret  : retrieval quality  (did last retrieval return relevant docs?)
  - θ_llm  : LLM reliability    (is the generated answer grounded in evidence?)
  - θ_noise: corpus noise level (are retrieved docs contradictory / off-topic?)
  - θ_diff : query difficulty   (is this a hard multi-hop / long-tail question?)

Each component: Beta(α, β) where α = prior_success + observed_successes,
                                   β = prior_failure + observed_failures.

Soft Beta-Bernoulli update: α += x, β += (1-x), for x ∈ [0,1].
The belief is updated at each step from observable signals (observations o_t).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class BeliefConfig:
    # Beta prior hyperparameters (α0, β0) for each θ component.
    # Values > 1 encode a weak informative prior; (1,1) = uniform / no prior.
    ret_alpha0: float = 2.0    # prior: retrieval tends to work
    ret_beta0: float = 1.0
    llm_alpha0: float = 3.0    # prior: LLM tends to be reliable when given evidence
    llm_beta0: float = 1.0
    noise_alpha0: float = 1.0  # prior: corpus is fairly clean
    noise_beta0: float = 3.0
    diff_alpha0: float = 1.0   # prior: questions are not extremely hard
    diff_beta0: float = 2.0

    # Thresholds for converting belief mean to a discrete regime label
    ret_low_thresh: float = 0.4    # below → "retrieval unreliable"
    ret_high_thresh: float = 0.7   # above → "retrieval reliable"
    diff_hard_thresh: float = 0.6  # above → "hard query"

    # E5-base-v2 empirical cosine similarity range used by obs_extractor
    # for score calibration. Stored here as single source of truth.
    e5_score_min: float = 0.50
    e5_score_max: float = 0.95


@dataclass
class BeliefState:
    """
    Maintains b_t(θ) = P(θ | history up to t) as four independent Beta distributions.

    Usage:
        belief = BeliefState()
        obs = extract_observation(docs, query, embedder, response_dict, split_queries)
        belief.update(**obs)
        prefix = belief_to_prompt_prefix(belief)   # inject into LLM prompt
        vec = belief.belief_vector                 # PPO policy state input
    """
    config: BeliefConfig = field(default_factory=BeliefConfig)

    # Beta sufficient statistics: (alpha, beta) per component.
    # alpha = prior_alpha + Σ successes; beta = prior_beta + Σ failures.
    _ret_alpha: float = field(init=False)
    _ret_beta: float = field(init=False)
    _llm_alpha: float = field(init=False)
    _llm_beta: float = field(init=False)
    _noise_alpha: float = field(init=False)
    _noise_beta: float = field(init=False)
    _diff_alpha: float = field(init=False)
    _diff_beta: float = field(init=False)

    # Step counter (number of update() calls since last reset)
    step: int = field(default=0, init=False)

    def __post_init__(self):
        self._init_from_config()

    def _init_from_config(self):
        """Initialize (or re-initialize) all sufficient statistics from priors."""
        c = self.config
        self._ret_alpha   = c.ret_alpha0
        self._ret_beta    = c.ret_beta0
        self._llm_alpha   = c.llm_alpha0
        self._llm_beta    = c.llm_beta0
        self._noise_alpha = c.noise_alpha0
        self._noise_beta  = c.noise_beta0
        self._diff_alpha  = c.diff_alpha0
        self._diff_beta   = c.diff_beta0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        retrieval_relevance: Optional[float] = None,    # composite retrieval quality (0–1)
        answer_consistency: Optional[float] = None,      # answer grounding score (0–1)
        doc_contradiction_rate: Optional[float] = None,  # topic drift / noise signal (0–1)
        query_hop_count: Optional[float] = None,         # difficulty signal, pre-converted (0–1)
    ):
        """
        Soft Beta-Bernoulli update: b_{t+1}(θ) ∝ Likelihood(o_t | θ) * b_t(θ)

        Each signal x ∈ [0,1] adds fractional counts:
            α += x,    β += (1 - x)

        Note: query_hop_count is now a pre-converted difficulty signal ∈ [0,1],
        computed by extract_query_hop_count(). The conversion is no longer done here.
        """
        self.step += 1

        # ── Input validation ─────────────────────────────────────────────────
        signals = {
            "retrieval_relevance": retrieval_relevance,
            "answer_consistency": answer_consistency,
            "doc_contradiction_rate": doc_contradiction_rate,
            "query_hop_count": query_hop_count,
        }
        for name, val in signals.items():
            if val is None:
                continue
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"BeliefState.update(): signal '{name}' = {val!r} is out of [0, 1]. "
                    "All signals must be in [0.0, 1.0] or None."
                )
            if val == 0.0 or val == 1.0:
                # Boundary values push the Beta distribution toward degenerate extremes
                print(
                    f"[BeliefState] Warning: signal '{name}' = {val} is at a boundary "
                    "value (0.0 or 1.0). This can over-update the Beta distribution."
                )

        # ── Beta-Bernoulli updates ───────────────────────────────────────────
        if retrieval_relevance is not None:
            self._ret_alpha += retrieval_relevance
            self._ret_beta  += (1.0 - retrieval_relevance)

        if answer_consistency is not None:
            self._llm_alpha += answer_consistency
            self._llm_beta  += (1.0 - answer_consistency)

        if doc_contradiction_rate is not None:
            # High contradiction rate → evidence of noisy corpus → α_noise increases
            self._noise_alpha += doc_contradiction_rate
            self._noise_beta  += (1.0 - doc_contradiction_rate)

        if query_hop_count is not None:
            # query_hop_count is already a difficulty signal ∈ [0,1] (from extractor)
            self._diff_alpha += query_hop_count
            self._diff_beta  += (1.0 - query_hop_count)

    # ------------------------------------------------------------------
    # Reset (for RL episode resets)
    # ------------------------------------------------------------------
    def reset(self):
        """
        Reset all Beta sufficient statistics to their prior values and
        zero the step counter. Use this between episodes in RL training.
        """
        self._init_from_config()
        self.step = 0

    # ------------------------------------------------------------------
    # Posterior mean  E[θ] = α / (α + β)
    # ------------------------------------------------------------------
    @property
    def ret_quality(self) -> float:
        return self._ret_alpha / (self._ret_alpha + self._ret_beta)

    @property
    def llm_reliability(self) -> float:
        return self._llm_alpha / (self._llm_alpha + self._llm_beta)

    @property
    def corpus_noise(self) -> float:
        return self._noise_alpha / (self._noise_alpha + self._noise_beta)

    @property
    def query_difficulty(self) -> float:
        return self._diff_alpha / (self._diff_alpha + self._diff_beta)

    # ------------------------------------------------------------------
    # Posterior uncertainty  Var[θ] = αβ / (α+β)²(α+β+1)
    # ------------------------------------------------------------------
    @staticmethod
    def _beta_variance(alpha: float, beta: float) -> float:
        total = alpha + beta
        return (alpha * beta) / (total * total * (total + 1))

    @property
    def ret_uncertainty(self) -> float:
        return self._beta_variance(self._ret_alpha, self._ret_beta)

    @property
    def llm_uncertainty(self) -> float:
        return self._beta_variance(self._llm_alpha, self._llm_beta)

    # ------------------------------------------------------------------
    # Belief vector — state input for PPO policy network
    # ------------------------------------------------------------------
    @property
    def belief_vector(self) -> List[float]:
        """
        Returns the raw Beta sufficient statistics as a flat vector.
        Shape: [ret_α, ret_β, llm_α, llm_β, noise_α, noise_β, diff_α, diff_β]
        This 8-dim vector is the observation fed into the PPO policy network.
        """
        return [
            self._ret_alpha,   self._ret_beta,
            self._llm_alpha,   self._llm_beta,
            self._noise_alpha, self._noise_beta,
            self._diff_alpha,  self._diff_beta,
        ]

    # ------------------------------------------------------------------
    # Discrete regime labels (for human-readable prompt injection)
    # ------------------------------------------------------------------
    @property
    def retrieval_regime(self) -> str:
        q = self.ret_quality
        if q < self.config.ret_low_thresh:
            return "unreliable"
        elif q > self.config.ret_high_thresh:
            return "reliable"
        return "uncertain"

    @property
    def is_hard_query(self) -> bool:
        return self.query_difficulty > self.config.diff_hard_thresh

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "ret_quality": round(self.ret_quality, 3),
            "ret_uncertainty": round(self.ret_uncertainty, 4),
            "llm_reliability": round(self.llm_reliability, 3),
            "llm_uncertainty": round(self.llm_uncertainty, 4),
            "corpus_noise": round(self.corpus_noise, 3),
            "query_difficulty": round(self.query_difficulty, 3),
            "retrieval_regime": self.retrieval_regime,
            "is_hard_query": self.is_hard_query,
            "belief_vector": [round(v, 4) for v in self.belief_vector],
        }

    def __repr__(self):
        d = self.to_dict()
        return (
            f"BeliefState(step={d['step']}, "
            f"ret={d['ret_quality']:.2f}±{d['ret_uncertainty']:.3f} [{d['retrieval_regime']}], "
            f"llm={d['llm_reliability']:.2f}, "
            f"noise={d['corpus_noise']:.2f}, "
            f"diff={d['query_difficulty']:.2f} [{'hard' if d['is_hard_query'] else 'easy'}])"
        )
