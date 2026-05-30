#!/usr/bin/env python3
"""Focused tests for context retrieval used by the LLM reasoner."""

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.llm_config import LLMConfig
from optimization.guider import Guider
from optimization.optimization_strategies import TCBOStrategy


class TestContextRetrieval(unittest.TestCase):
    def test_plan_aware_selection_removes_duplicate_plan_ids(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        observations = [
            ([0.10, 0.10], 1000, "plan_a"),
            ([0.11, 0.11], 1010, "plan_a"),
            ([0.20, 0.20], 900, "plan_b"),
            ([0.30, 0.30], 800, "plan_c"),
        ]
        for vector, perf, plan_id in observations:
            guider.record_observation(vector, perf, plan_id=plan_id)

        similar = guider.get_similar_observations([0.105, 0.105], 3)
        self.assertEqual(len(similar), 3)
        self.assertEqual(similar[0][0], [0.10, 0.10])
        self.assertNotIn(([0.11, 0.11], 1010), similar)

    def test_tcbo_context_uses_raw_latency_not_internal_objective(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            strategy="tcbo",
            timeout_threshold=2_000.0,
        )
        guider.record_observation([0.10, 0.10], 1_000.0, plan_id="plan_a")
        guider.record_observation([0.20, 0.20], 1_500.0, plan_id="plan_b")

        similar = guider.get_similar_observations([0.11, 0.11], 2)

        self.assertEqual([score for _, score in similar], [1_000.0, 1_500.0])
        self.assertEqual(guider.strategy.Y, [-1_000.0, -1_500.0])

    def test_tcbo_constraint_preserves_timeout_magnitude(self) -> None:
        strategy = TCBOStrategy(dimension=2, timeout_threshold=2_000.0)

        strategy.tell([0.1, 0.1], 2_500.0)
        strategy.tell([0.2, 0.2], 2_000.0, is_timeout=True)

        self.assertEqual(strategy.C[0], 500.0)
        self.assertGreater(strategy.C[1], 0.0)

    def test_tcbo_timeout_label_is_explicit_not_latency_equality(self) -> None:
        strategy = TCBOStrategy(dimension=2, timeout_threshold=2_000.0)

        strategy.tell([0.1, 0.1], 2_000.0, is_timeout=False)

        self.assertEqual(strategy.C, [0.0])
        self.assertEqual(strategy.timeout_labels, [0])
        self.assertEqual(strategy.objective_observed, [True])

    def test_tcbo_tracks_timeout_risk_separately_from_latency_margin(self) -> None:
        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            num_trust_regions=3,
            risk_threshold=0.05,
        )

        strategy.tell([0.1, 0.1], 1_000.0)
        strategy.tell([0.9, 0.9], 2_500.0)

        self.assertEqual(len(strategy.trust_regions), 3)
        self.assertEqual(strategy.timeout_labels, [0, 1])
        self.assertAlmostEqual(strategy.timeout_constraints[0], -0.05)
        self.assertAlmostEqual(strategy.timeout_constraints[1], 0.95)
        self.assertEqual(strategy.C, [-1_000.0, 500.0])
        self.assertEqual(strategy.objective_observed, [True, False])

    def test_guider_respects_zero_warm_start_rounds(self) -> None:
        guider = Guider(
            warm_start_rounds=0,
            knob_names=["k1"],
            llm_config=LLMConfig(enabled=False),
        )

        self.assertEqual(guider.warm_start_rounds, 0)
        self.assertEqual(guider.warm_start_sampling(), [])

        default_guider = Guider(
            knob_names=["k1"],
            llm_config=LLMConfig(enabled=False),
        )

        self.assertEqual(default_guider.warm_start_rounds, 6)

    def test_tcbo_trust_region_update_uses_paper_multipliers(self) -> None:
        strategy = TCBOStrategy(dimension=20, timeout_threshold=2_000.0)
        state = strategy.trust_regions[0]

        self.assertEqual(state.success_tolerance, 3)
        self.assertEqual(state.failure_tolerance, 3)

        state.length = 0.2
        state.success_counter = 2
        state.best_value = -1_000.0
        state.best_constraint_value = -1_000.0
        strategy._update_state(
            torch.tensor([[-900.0]], dtype=strategy.dtype, device=strategy.device),
            torch.tensor([[-1_000.0]], dtype=strategy.dtype, device=strategy.device),
            state=state,
        )

        self.assertAlmostEqual(state.length, 0.4)
        self.assertEqual(state.success_counter, 0)

        state.failure_counter = 2
        state.success_counter = 0
        state.best_value = -800.0
        strategy._update_state(
            torch.tensor([[-900.0]], dtype=strategy.dtype, device=strategy.device),
            torch.tensor([[-1_000.0]], dtype=strategy.dtype, device=strategy.device),
            state=state,
        )

        self.assertAlmostEqual(state.length, 0.2)
        self.assertEqual(state.failure_counter, 0)

    def test_tcbo_trust_region_counters_are_consecutive_events(self) -> None:
        strategy = TCBOStrategy(dimension=20, timeout_threshold=2_000.0)
        state = strategy.trust_regions[0]
        state.length = 0.2
        state.best_value = -1_000.0
        state.best_constraint_value = -1_000.0

        strategy._update_state(
            torch.tensor([[-900.0]], dtype=strategy.dtype, device=strategy.device),
            torch.tensor([[-1_000.0]], dtype=strategy.dtype, device=strategy.device),
            state=state,
        )
        self.assertEqual(state.success_counter, 1)
        self.assertEqual(state.failure_counter, 0)
        self.assertAlmostEqual(state.length, 0.2)

        strategy._update_state(
            torch.tensor([[-950.0]], dtype=strategy.dtype, device=strategy.device),
            torch.tensor([[-1_000.0]], dtype=strategy.dtype, device=strategy.device),
            state=state,
        )
        self.assertEqual(state.success_counter, 0)
        self.assertEqual(state.failure_counter, 1)
        self.assertAlmostEqual(state.length, 0.2)

        state.best_constraint_value = 500.0
        strategy._update_state(
            torch.tensor([[-float("inf")]], dtype=strategy.dtype, device=strategy.device),
            torch.tensor([[400.0]], dtype=strategy.dtype, device=strategy.device),
            state=state,
        )
        self.assertEqual(state.success_counter, 1)
        self.assertEqual(state.failure_counter, 0)

    def test_tcbo_trust_region_restart_triggers_below_length_min(self) -> None:
        strategy = TCBOStrategy(dimension=20, timeout_threshold=2_000.0)
        state = strategy.trust_regions[0]
        state.length = state.length_min * 1.5
        state.failure_counter = 2
        state.best_value = -800.0
        state.best_constraint_value = -1_000.0

        strategy._update_state(
            torch.tensor([[-900.0]], dtype=strategy.dtype, device=strategy.device),
            torch.tensor([[-1_000.0]], dtype=strategy.dtype, device=strategy.device),
            state=state,
        )

        self.assertLess(state.length, state.length_min)
        self.assertTrue(state.restart_triggered)

    def test_tcbo_admission_reject_updates_safety_not_objective(self) -> None:
        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            num_trust_regions=2,
        )
        strategy.tell([0.1, 0.1], 1_000.0)
        strategy.tell([0.2, 0.2], 900.0)
        strategy.tell_admission_rejection([0.9, 0.9], estimated_latency=3_000.0)

        self.assertEqual(strategy.Y, [-1_000.0, -900.0, -3_000.0])
        self.assertEqual(strategy.timeout_labels, [0, 0, 1])
        self.assertEqual(strategy.objective_observed, [True, True, False])
        self.assertEqual(strategy._objective_indices_for_region(strategy.trust_regions[0]), [0, 1])

    def test_tcbo_admission_reject_updates_trust_region_state(self) -> None:
        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            num_trust_regions=1,
        )
        state = strategy.trust_regions[0]
        state.center = [0.5, 0.5]
        state.best_constraint_value = 100.0
        state.failure_counter = 2

        strategy.tell_admission_rejection([0.5, 0.5], estimated_latency=2_500.0)

        self.assertAlmostEqual(state.length, 0.1)
        self.assertEqual(state.failure_counter, 0)
        self.assertEqual(strategy.Y, [-2_500.0])
        self.assertEqual(strategy.objective_observed, [False])
        self.assertEqual(strategy.timeout_labels, [1])

    def test_guider_excludes_admission_rejects_from_llm_context(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            strategy="tcbo",
            timeout_threshold=2_000.0,
        )
        guider.record_observation([0.1, 0.1], 1_000.0, plan_id="plan_a")
        guider.record_admission_rejection(
            [0.11, 0.11],
            estimated_latency=3_000.0,
            plan_fingerprint="plan_bad",
        )
        guider.record_observation([0.2, 0.2], 900.0, plan_id="plan_b")

        similar = guider.get_similar_observations([0.11, 0.11], 3)

        self.assertEqual([vector for vector, _ in similar], [[0.1, 0.1], [0.2, 0.2]])
        self.assertEqual(guider.strategy.objective_observed, [True, False, True])

    def test_guider_excludes_timeout_censored_observations_from_llm_context(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            strategy="tcbo",
            timeout_threshold=2_000.0,
        )
        guider.record_observation([0.1, 0.1], 1_000.0, plan_id="plan_a")
        guider.record_observation([0.11, 0.11], 2_500.0, plan_id="plan_timeout")
        guider.record_observation([0.2, 0.2], 900.0, plan_id="plan_b")

        similar = guider.get_similar_observations([0.11, 0.11], 3)

        self.assertEqual([vector for vector, _ in similar], [[0.1, 0.1], [0.2, 0.2]])
        self.assertEqual(guider.strategy.objective_observed, [True, False, True])

    def test_guider_passes_configured_tcbo_controls(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            strategy="tcbo",
            timeout_threshold=2_000.0,
            tcbo_num_trust_regions=2,
            tcbo_risk_threshold=0.2,
            tcbo_candidate_count=17,
        )

        self.assertEqual(guider.strategy.num_trust_regions, 2)
        self.assertAlmostEqual(guider.strategy.risk_threshold, 0.2)
        self.assertEqual(guider.strategy.n_candidates, 17)

    def test_tcbo_timeout_risk_model_predicts_probability(self) -> None:
        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            timeout_classifier_train_steps=2,
        )
        strategy.tell([0.1, 0.1], 1_000.0)
        strategy.tell([0.9, 0.9], 2_500.0)

        timeout_model = strategy._timeout_risk_model()

        self.assertIsNotNone(timeout_model)
        probabilities = timeout_model.predict_timeout_probability(
            torch.tensor([[0.1, 0.1], [0.9, 0.9]], dtype=strategy.dtype, device=strategy.device)
        )
        self.assertEqual(probabilities.shape, torch.Size([2]))
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.all(probabilities <= 1.0))
        sampled_probabilities = timeout_model.sample_timeout_probability(
            torch.tensor([[0.1, 0.1], [0.9, 0.9]], dtype=strategy.dtype, device=strategy.device)
        )
        self.assertEqual(sampled_probabilities.shape, torch.Size([2]))
        self.assertTrue(torch.all(sampled_probabilities >= 0.0))
        self.assertTrue(torch.all(sampled_probabilities <= 1.0))

    def test_tcbo_candidate_pool_uses_timeout_probability_threshold(self) -> None:
        class FakeTimeoutModel:
            def predict_timeout_probability(self, candidates):
                return candidates[:, 0]

        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            risk_threshold=0.2,
        )
        candidates = torch.tensor(
            [[0.1, 0.1], [0.25, 0.2], [0.05, 0.9]],
            dtype=strategy.dtype,
            device=strategy.device,
        )

        safe_pool = strategy._safe_candidate_pool(candidates, FakeTimeoutModel())

        self.assertEqual(safe_pool.tolist(), [[0.1, 0.1], [0.05, 0.9]])

    def test_tcbo_candidate_pool_keeps_least_risky_when_all_candidates_unsafe(self) -> None:
        class FakeTimeoutModel:
            def predict_timeout_probability(self, candidates):
                return candidates[:, 0] + 0.5

        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            risk_threshold=0.05,
        )
        candidates = torch.tensor(
            [[0.4, 0.1], [0.1, 0.2], [0.3, 0.9]],
            dtype=strategy.dtype,
            device=strategy.device,
        )

        fallback_pool = strategy._safe_candidate_pool(candidates, FakeTimeoutModel())

        self.assertEqual(fallback_pool[0].tolist(), [0.1, 0.2])

    def test_tcbo_ask_returns_normalized_candidate_after_mixed_feedback(self) -> None:
        strategy = TCBOStrategy(
            dimension=2,
            timeout_threshold=2_000.0,
            num_trust_regions=3,
        )
        for vector, perf in [
            ([0.1, 0.1], 1_000.0),
            ([0.2, 0.2], 1_200.0),
            ([0.8, 0.8], 2_500.0),
            ([0.7, 0.75], 1_300.0),
        ]:
            strategy.tell(vector, perf)

        candidate = strategy.ask()

        self.assertEqual(len(candidate), 2)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in candidate))

    def test_tcbo_gaussian_copula_transform_preserves_objective_order(self) -> None:
        strategy = TCBOStrategy(dimension=2, timeout_threshold=2_000.0)
        values = torch.tensor([[-3.0], [-1.0], [-2.0]], dtype=strategy.dtype)

        transformed = strategy._gaussian_copula_transform(values)

        self.assertEqual(transformed.shape, values.shape)
        self.assertTrue(torch.all(torch.isfinite(transformed)))
        self.assertEqual(torch.argsort(values.squeeze(-1)).tolist(), torch.argsort(transformed.squeeze(-1)).tolist())

    def test_context_selection_follows_eq9_nearest_distinct_plans(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        observations = [
            ([0.48, 0.48], 1000, "plan_a"),
            ([0.49, 0.49], 990, "plan_a"),
            ([0.52, 0.52], 960, "plan_b"),
            ([0.60, 0.60], 960, "plan_c"),
            ([0.80, 0.80], 700, "plan_d"),
        ]
        for vector, perf, plan_id in observations:
            guider.record_observation(vector, perf, plan_id=plan_id)

        unique_indices = guider._unique_observation_indices_by_vector([0.50, 0.50])
        selected_indices = guider._select_context_indices([0.50, 0.50], unique_indices, 3)
        selected_vectors = [guider.strategy.X[idx] for idx in selected_indices]

        self.assertEqual(selected_vectors, [[0.49, 0.49], [0.52, 0.52], [0.60, 0.60]])

    def test_guider_fallback_preserves_xbo_when_reasoner_fails(self) -> None:
        class FailingReasoner:
            def __init__(self, config=None):
                pass

            def recommend_next_configs(self, *args, **kwargs):
                raise RuntimeError("llm unavailable")

        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.1, 0.1], 1000.0, plan_id="plan_a")
        guider.record_observation([0.2, 0.2], 900.0, plan_id="plan_b")
        guider.suggest_vector = lambda: [0.7, 0.7]

        with patch("optimization.guider.ConfigurationReasoner", FailingReasoner):
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=3,
            )

        self.assertEqual(len(vectors), 4)
        self.assertEqual(guider.last_candidate_source, "VanillaGP Sampling")
        self.assertIn([0.7, 0.7], vectors)
        self.assertTrue(all(len(vector) == 2 for vector in vectors))
        self.assertTrue(all(0.0 <= value <= 1.0 for vector in vectors for value in vector))

    def test_guider_skips_reasoner_without_remote_llm_config(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="", base_url=""),
        )
        guider.suggest_vector = lambda: [0.4, 0.6]

        with patch("optimization.guider.ConfigurationReasoner") as reasoner_class:
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=2,
            )

        reasoner_class.assert_not_called()
        self.assertEqual(guider.last_candidate_source, "VanillaGP Sampling")
        self.assertEqual(len(vectors), 3)
        self.assertIn([0.4, 0.6], vectors)

    def test_guider_skips_reasoner_without_true_objective_context(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            strategy="tcbo",
            timeout_threshold=100.0,
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.8, 0.8], 100.0, plan_id="plan_timeout")
        guider.suggest_vector = lambda: [0.4, 0.6]

        with patch("optimization.guider.ConfigurationReasoner") as reasoner_class:
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=2,
            )

        reasoner_class.assert_not_called()
        self.assertEqual(guider.last_candidate_source, "TCBO Sampling")
        self.assertEqual(len(vectors), 3)
        self.assertIn([0.4, 0.6], vectors)

    def test_guider_skips_reasoner_when_llm_disabled(self) -> None:
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(enabled=False, api_key="test-key"),
        )
        guider.suggest_vector = lambda: [0.2, 0.8]

        with patch("optimization.guider.ConfigurationReasoner") as reasoner_class:
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=2,
            )

        reasoner_class.assert_not_called()
        self.assertEqual(guider.last_candidate_source, "VanillaGP Sampling")
        self.assertEqual(len(vectors), 3)
        self.assertIn([0.2, 0.8], vectors)

    def test_sampling_retry_preserves_xbo_without_reasoner(self) -> None:
        guider = Guider(warm_start_rounds=3, knob_names=["k1", "k2"])
        guider.suggest_vector = lambda: [0.3, 0.7]

        with patch("optimization.guider.ConfigurationReasoner") as reasoner_class:
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=2,
                batch=2,
            )

        reasoner_class.assert_not_called()
        self.assertEqual(guider.last_candidate_source, "VanillaGP Sampling")
        self.assertEqual(len(vectors), 3)
        self.assertIn([0.3, 0.7], vectors)
        self.assertTrue(all(0.0 <= value <= 1.0 for vector in vectors for value in vector))

    def test_rejection_aware_batch_includes_guider_candidate(self) -> None:
        class RejectionAwareReasoner:
            def __init__(self, config=None):
                pass

            def recommend_next_configs_after_rejection(self, *args, **kwargs):
                return [[0.2, 0.2]]

        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.1, 0.1], 1000.0, plan_id="plan_a")
        guider.record_observation([0.3, 0.3], 900.0, plan_id="plan_b")
        guider.suggest_vector = lambda: [0.7, 0.7]

        with patch("optimization.guider.ConfigurationReasoner", RejectionAwareReasoner):
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=1,
                last_failed_vectors=[[0.9, 0.9]],
                batch=1,
            )

        self.assertEqual(vectors, [[0.2, 0.2], [0.7, 0.7]])
        self.assertEqual(guider.last_candidate_source, "VanillaGP+Reasoner Prompt Optimization")

    def test_reasoner_batch_truncates_to_requested_size_before_xbo(self) -> None:
        class OversizedReasoner:
            def __init__(self, config=None):
                pass

            def recommend_next_configs(self, *args, **kwargs):
                return [[0.2, 0.1], [0.3, 0.4], [0.5, 0.6]]

        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.1, 0.1], 1000.0, plan_id="plan_a")
        guider.record_observation([0.2, 0.2], 900.0, plan_id="plan_b")
        guider.suggest_vector = lambda: [0.7, 0.7]

        with patch("optimization.guider.ConfigurationReasoner", OversizedReasoner):
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=2,
            )

        self.assertEqual(vectors, [[0.2, 0.1], [0.3, 0.4], [0.7, 0.7]])
        self.assertEqual(guider.last_candidate_source, "VanillaGP+Reasoner")

    def test_reasoner_batch_defaults_non_finite_coordinates_before_xbo(self) -> None:
        class NonFiniteReasoner:
            def __init__(self, config=None):
                pass

            def recommend_next_configs(self, *args, **kwargs):
                return [[float("nan"), float("inf")]]

        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.1, 0.1], 1000.0, plan_id="plan_a")
        guider.record_observation([0.2, 0.2], 900.0, plan_id="plan_b")
        guider.suggest_vector = lambda: [0.7, 0.7]

        with patch("optimization.guider.ConfigurationReasoner", NonFiniteReasoner):
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=1,
            )

        self.assertEqual(vectors, [[0.5, 0.5], [0.7, 0.7]])
        self.assertEqual(guider.last_candidate_source, "VanillaGP+Reasoner")

    def test_incomplete_reasoner_batch_falls_back_to_sampling(self) -> None:
        class IncompleteReasoner:
            def __init__(self, config=None):
                pass

            def recommend_next_configs(self, *args, **kwargs):
                return [[0.2, 0.2], [0.2, 0.2], [0.3]]

        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.1, 0.1], 1000.0, plan_id="plan_a")
        guider.record_observation([0.4, 0.4], 900.0, plan_id="plan_b")
        guider.suggest_vector = lambda: [0.7, 0.7]
        guider.lhs_fallback_sampling = lambda batch: [[0.11, 0.22], [0.33, 0.44]]

        with patch("optimization.guider.ConfigurationReasoner", IncompleteReasoner):
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=2,
            )

        self.assertEqual(vectors, [[0.11, 0.22], [0.33, 0.44], [0.7, 0.7]])
        self.assertEqual(guider.last_candidate_source, "VanillaGP Sampling")

    def test_reasoner_batch_deduplicates_guider_candidate(self) -> None:
        guider = Guider(warm_start_rounds=3, knob_names=["k1", "k2"])

        merged = guider._merge_reasoner_batch_with_guider_point(
            [[0.2, 0.2], [0.7, 0.7]],
            [0.7, 0.7],
        )

        self.assertEqual(merged, [[0.2, 0.2], [0.7, 0.7]])

    def test_guider_passes_xbo_to_reasoner(self) -> None:
        class CapturingReasoner:
            guider_vectors = []

            def __init__(self, config=None):
                pass

            def recommend_next_configs(self, *args, **kwargs):
                self.guider_vectors.append(kwargs["guider_vector"])
                return [[0.2, 0.2]]

        CapturingReasoner.guider_vectors = []
        guider = Guider(
            warm_start_rounds=3,
            knob_names=["k1", "k2"],
            llm_config=LLMConfig(api_key="test-key"),
        )
        guider.record_observation([0.1, 0.1], 1000.0, plan_id="plan_a")
        guider.record_observation([0.3, 0.3], 900.0, plan_id="plan_b")
        guider.suggest_vector = lambda: [0.7, 0.7]

        with patch("optimization.guider.ConfigurationReasoner", CapturingReasoner):
            vectors = guider.get_next_points(
                sql_path="unused.sql",
                topk=2,
                try_number=0,
                batch=1,
            )

        self.assertEqual(CapturingReasoner.guider_vectors, [[0.7, 0.7]])
        self.assertEqual(vectors, [[0.2, 0.2], [0.7, 0.7]])
        self.assertEqual(guider.last_candidate_source, "VanillaGP+Reasoner")


if __name__ == "__main__":
    unittest.main(verbosity=2)
