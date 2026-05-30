from typing import List, Optional

import torch
from pyDOE import lhs

from llm.reasoner import ConfigurationReasoner
from llm.llm_config import LLMConfig
from optimization.optimization_strategies import VanillaGPStrategy, TCBOStrategy
from util.knob_space import KnobSpace
from util.logger import logger


class Guider:
    """Bayesian optimization Guider from OBELISK Section 6."""

    def __init__(
        self,
        knob_space: KnobSpace = None,
        warm_start_rounds: int = None,
        knob_names: List[str] = None,
        strategy: str = "vanilla_gp",
        timeout_threshold: float = None,
        llm_config: LLMConfig | None = None,
        tcbo_num_trust_regions: int = 4,
        tcbo_risk_threshold: float = 0.05,
        tcbo_candidate_count: int = 2000,
        warm_start_times: int = None,
    ):
        if knob_space is not None:
            self.knob_space = knob_space
        elif knob_names is not None:
            self.knob_space = KnobSpace(knob_names)
        else:
            raise ValueError("Either knob_space or knob_names must be provided")
            
        if warm_start_rounds is None:
            warm_start_rounds = warm_start_times
        if warm_start_rounds is None:
            warm_start_rounds = 6
        warm_start_rounds = int(warm_start_rounds)
        if warm_start_rounds < 0:
            raise ValueError("warm_start_rounds must be non-negative")

        self.warm_start_rounds = warm_start_rounds
        self.warm_start_times = self.warm_start_rounds
        if strategy == "tcbo":
            if timeout_threshold is None:
                raise ValueError("timeout_threshold is required for TCBO strategy")
            self.strategy = TCBOStrategy(
                self.knob_space.dimension,
                timeout_threshold,
                num_trust_regions=tcbo_num_trust_regions,
                risk_threshold=tcbo_risk_threshold,
                n_candidates=tcbo_candidate_count,
            )
            logger.info(
                "Using TCBO strategy: timeout=%.2fs trust_regions=%d risk_threshold=%.3f candidates=%d",
                timeout_threshold / 1000,
                self.strategy.num_trust_regions,
                self.strategy.risk_threshold,
                self.strategy.n_candidates,
            )
        else:
            self.strategy = VanillaGPStrategy(self.knob_space.dimension)
            logger.info("Using VanillaGP strategy")
        self.llm_config = llm_config or LLMConfig.from_app_config()
        self.observed_vectors: List[List[float]] = []
        self.observed_latencies: List[float] = []
        self.plan_fingerprints: List[str] = []
        self.plan_ids = self.plan_fingerprints
        self.last_candidate_source = ""

    def warm_start_sampling(self) -> List[List[float]]:
        if self.warm_start_rounds <= 0:
            return []
        if self.knob_space.dimension == 0:
            return [[] for _ in range(self.warm_start_rounds)]

        engine = torch.quasirandom.SobolEngine(
            dimension=self.knob_space.dimension,
            scramble=True,
        )
        return engine.draw(self.warm_start_rounds).tolist()

    def lhs_fallback_sampling(self, batch: int) -> List[List[float]]:
        if batch <= 0:
            return []
        if self.knob_space.dimension == 0:
            return [[] for _ in range(batch)]
        lhs_samples = lhs(self.knob_space.dimension, samples=batch)
        return [sample.tolist() for sample in lhs_samples]

    def suggest_vector(self) -> List[float]:
        return self.strategy.ask()

    def record_observation(
        self,
        vector: List[float],
        perf: float,
        plan_id: Optional[str] = None,
        plan_fingerprint: Optional[str] = None,
        is_true_objective: bool | None = None,
        is_timeout: bool | None = None,
    ) -> None:
        self.strategy.tell(vector, perf, is_timeout=is_timeout)
        if is_true_objective is None:
            is_true_objective = self._latest_strategy_objective_observed()
        if is_true_objective:
            self.observed_vectors.append(vector)
            self.observed_latencies.append(perf)
            fingerprint = plan_fingerprint if plan_fingerprint is not None else plan_id
            self.plan_fingerprints.append(fingerprint or "")

    def _latest_strategy_objective_observed(self) -> bool:
        objective_observed = getattr(self.strategy, "objective_observed", None)
        if not objective_observed:
            return True
        return bool(objective_observed[-1])

    def record_admission_rejection(
        self,
        vector: List[float],
        estimated_latency: float | None = None,
        plan_fingerprint: Optional[str] = None,
    ) -> None:
        """Record Evaluator safety feedback without treating it as y."""
        self.strategy.tell_admission_rejection(vector, estimated_latency)

    @staticmethod
    def _vector_signature(vector: List[float]) -> tuple[float, ...]:
        return tuple(round(v, 8) for v in vector)

    def _get_plan_fingerprint(self, idx: int) -> str:
        if idx < len(self.plan_fingerprints):
            return self.plan_fingerprints[idx] or ""
        return ""

    def _get_plan_id(self, idx: int) -> str:
        """Backward-compatible alias for plan-fingerprint lookup."""
        return self._get_plan_fingerprint(idx)

    def _unique_observation_indices_by_vector(self, vector: List[float]) -> List[int]:
        if not self.observed_vectors:
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_tensor = torch.tensor(vector, dtype=torch.double, device=device)
        x_tensor = torch.tensor(self.observed_vectors, dtype=torch.double, device=device)
        distances = torch.norm(x_tensor - query_tensor, dim=1)
        sorted_indices = torch.argsort(distances).tolist()

        unique_indices = []
        seen_vectors = set()
        for idx in sorted_indices:
            signature = self._vector_signature(self.observed_vectors[idx])
            if signature in seen_vectors:
                continue
            seen_vectors.add(signature)
            unique_indices.append(idx)
        return unique_indices

    def _select_context_indices(self, _vector: List[float], candidate_indices: List[int], k: int) -> List[int]:
        """Select nearest observations while enforcing distinct plans, as in Eq. (9)."""
        if not candidate_indices or k <= 0:
            return []

        selected = []
        seen_plan_fingerprints = set()
        for idx in candidate_indices:
            if len(selected) >= k:
                break
            plan_fingerprint = self._get_plan_fingerprint(idx)
            if plan_fingerprint and plan_fingerprint in seen_plan_fingerprints:
                continue
            selected.append(idx)
            if plan_fingerprint:
                seen_plan_fingerprints.add(plan_fingerprint)

        return selected

    def get_similar_observations(self, vector: List[float], k: int) -> List[tuple[List[float], float]]:
        if not self.observed_vectors:
            return []

        unique_indices = self._unique_observation_indices_by_vector(vector)
        selected_indices = self._select_context_indices(vector, unique_indices, k)

        results = []
        for idx in selected_indices:
            if idx < len(self.observed_latencies):
                results.append((self.observed_vectors[idx], self.observed_latencies[idx]))
        return results

    def _merge_reasoner_batch_with_guider_point(
        self,
        reasoner_vectors: List[List[float]],
        guider_vector: List[float],
        batch: int | None = None,
    ) -> List[List[float]]:
        """Return X union {xBO}, preserving valid Reasoner order first."""
        merged = []
        seen = set()
        candidates = list(reasoner_vectors or [])
        if batch is None:
            candidates.append(guider_vector)
        else:
            batch = int(batch)
            if batch < 0:
                raise ValueError("batch must be non-negative")
            if batch == 0:
                clamped = self.knob_space.clamp_vector(guider_vector)
                return [clamped] if len(clamped) == self.knob_space.dimension else []
        for candidate in candidates:
            if len(candidate) != self.knob_space.dimension:
                continue
            clamped = self.knob_space.clamp_vector(candidate)
            signature = self._vector_signature(clamped)
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(clamped)
            if batch is not None and len(merged) >= batch:
                break
        if batch is not None and len(merged) < batch:
            return []
        if batch is not None:
            guider_candidate = self.knob_space.clamp_vector(guider_vector)
            if len(guider_candidate) == self.knob_space.dimension:
                signature = self._vector_signature(guider_candidate)
                if signature not in seen:
                    merged.append(guider_candidate)
        return merged

    def _reasoner_batch_with_guider_point(
        self,
        reasoner_vectors: List[List[float]],
        guider_vector: List[float],
        batch: int,
    ) -> List[List[float]]:
        """Require a complete Reasoner batch X before adding xBO."""
        batch = int(batch)
        if batch < 0:
            raise ValueError("batch must be non-negative")
        return self._merge_reasoner_batch_with_guider_point(
            reasoner_vectors,
            guider_vector,
            batch=batch,
        )

    def _sampling_batch_with_guider_point(
        self,
        guider_vector: List[float],
        batch: int,
    ) -> List[List[float]]:
        """Use design-space sampling while preserving the Guider xBO point."""
        return self._merge_reasoner_batch_with_guider_point(
            self.lhs_fallback_sampling(batch),
            guider_vector,
        )

    def _candidate_source_name(self, try_number: int, used_reasoner: bool) -> str:
        strategy_name = type(self.strategy).__name__.replace("Strategy", "")
        if not used_reasoner:
            return f"{strategy_name} Sampling"
        if try_number == 0:
            return f"{strategy_name}+Reasoner"
        return f"{strategy_name}+Reasoner Prompt Optimization"

    def get_next_points(
        self,
        sql_path: str,
        topk: int,
        try_number: int = 0,
        last_failed_vectors: List[List[float]] | None = None,
        batch: int = 1,
    ) -> List[List[float]]:
        last_failed_vectors = last_failed_vectors or []
        batch = int(batch)
        if batch < 0:
            raise ValueError("batch must be non-negative")
        vector = self.suggest_vector()

        similar_observations = self.get_similar_observations(vector, topk)
        if not similar_observations:
            self.last_candidate_source = self._candidate_source_name(
                try_number,
                used_reasoner=False,
            )
            logger.info(
                "No true objective observations are available for Reasoner context; "
                "using Latin hypercube sampling plus xBO"
            )
            return self._sampling_batch_with_guider_point(vector, batch)

        if not self.llm_config.can_call_remote():
            self.last_candidate_source = self._candidate_source_name(
                try_number,
                used_reasoner=False,
            )
            logger.info(
                "LLM Reasoner is disabled or has no remote endpoint/key; "
                "using Latin hypercube sampling plus xBO"
            )
            return self._sampling_batch_with_guider_point(vector, batch)

        reasoner = ConfigurationReasoner(config=self.llm_config)
        try:
            if try_number == 0:
                next_vectors = reasoner.recommend_next_configs(
                    observations=similar_observations,
                    sql_path=sql_path,
                    knob_names=self.knob_space.knob_names,
                    batch=batch,
                    guider_vector=vector,
                )
                next_vectors = self._reasoner_batch_with_guider_point(
                    next_vectors,
                    vector,
                    batch,
                )
                if not next_vectors:
                    raise ValueError("Reasoner returned fewer than batch distinct valid vectors")
                self.last_candidate_source = self._candidate_source_name(
                    try_number,
                    used_reasoner=True,
                )
            else:
                next_vectors = reasoner.recommend_next_configs_after_rejection(
                    observations=similar_observations,
                    sql_path=sql_path,
                    last_failed_vectors=last_failed_vectors[-5:],
                    knob_names=self.knob_space.knob_names,
                    batch=batch,
                    guider_vector=vector,
                )
                next_vectors = self._reasoner_batch_with_guider_point(
                    next_vectors,
                    vector,
                    batch,
                )
                if not next_vectors:
                    raise ValueError("Reasoner returned fewer than batch distinct valid vectors")
                self.last_candidate_source = self._candidate_source_name(
                    try_number,
                    used_reasoner=True,
                )
        except Exception as exc:
            logger.warning(
                "Reasoner failed; falling back to Latin hypercube sampling plus xBO: %s",
                exc,
            )
            next_vectors = self._sampling_batch_with_guider_point(vector, batch)
            self.last_candidate_source = self._candidate_source_name(
                try_number,
                used_reasoner=False,
            )

        valid_vectors = [
            self.knob_space.clamp_vector(candidate)
            for candidate in next_vectors
            if len(candidate) == self.knob_space.dimension
        ]
        if valid_vectors:
            return valid_vectors

        self.last_candidate_source = self._candidate_source_name(
            try_number,
            used_reasoner=False,
        )
        return self._sampling_batch_with_guider_point(vector, batch)


BayesianGuider = Guider
Tuner = Guider
