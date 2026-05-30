#!/usr/bin/env python3
"""Unit tests for LLM client configuration and knob naming."""

# ruff: noqa: E402

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.llm import ConfigurationReasoner, LLM
from llm.llm_config import LLMConfig
from llm.promptopt.prompt_optimizer import PromptOptimizer
from llm.prompts import normal_config_prompt


class TestLLMConfig(unittest.TestCase):
    def test_prompt_optimizer_uses_openai_compatible_chat_completion(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="optimized prompt")
                        )
                    ]
                )

        class FakeOpenAI:
            init_kwargs = {}
            completions = FakeCompletions()

            def __init__(self, **kwargs) -> None:
                FakeOpenAI.init_kwargs = kwargs
                self.chat = SimpleNamespace(
                    completions=FakeOpenAI.completions
                )

        prompt_pool_path = SRC_DIR / "llm" / "promptopt" / "prompt_pool.yaml"
        with patch("llm.promptopt.prompt_optimizer.OpenAI", FakeOpenAI):
            optimizer = PromptOptimizer(
                str(prompt_pool_path),
                model_name="provider-chat",
                api_key="test-key",
                base_url="https://llm.example.com/v1",
                temperature=0.3,
                max_retries=2,
                max_new_tokens=321,
                top_p=0.4,
            )
            response = optimizer._chat_completion("user task", "system role")

        self.assertEqual(response, "optimized prompt")
        self.assertEqual(FakeOpenAI.init_kwargs["api_key"], "test-key")
        self.assertEqual(FakeOpenAI.init_kwargs["base_url"], "https://llm.example.com/v1")
        self.assertEqual(FakeOpenAI.init_kwargs["max_retries"], 2)
        call = FakeOpenAI.completions.calls[-1]
        self.assertEqual(call["model"], "provider-chat")
        self.assertEqual(call["temperature"], 0.3)
        self.assertEqual(call["top_p"], 0.4)
        self.assertEqual(call["max_tokens"], 321)
        self.assertEqual(call["messages"][0]["role"], "system")
        self.assertEqual(call["messages"][1]["role"], "user")

    def test_chat_model_uses_openai_compatible_base_url(self) -> None:
        reasoner = ConfigurationReasoner(
            LLMConfig(
                model_name="provider-chat",
                api_key="test-key",
                base_url="https://llm.example.com/v1",
            )
        )

        model = reasoner._create_model()

        self.assertEqual(str(model.openai_api_base).rstrip("/"), "https://llm.example.com/v1")

    def test_reasoner_chat_model_disables_provider_retry(self) -> None:
        class FakeChatOpenAI:
            init_kwargs = {}

            def __init__(self, **kwargs) -> None:
                FakeChatOpenAI.init_kwargs = kwargs

        reasoner = ConfigurationReasoner(
            LLMConfig(api_key="test-key", max_retries=7, max_new_tokens=321)
        )

        with patch("llm.reasoner.ChatOpenAI", FakeChatOpenAI):
            reasoner._create_model()

        self.assertEqual(FakeChatOpenAI.init_kwargs["max_retries"], 0)
        self.assertEqual(FakeChatOpenAI.init_kwargs["max_completion_tokens"], 321)

    def test_reasoner_openai_compatible_retry_budget_is_not_multiplied(self) -> None:
        class FakeCompletions:
            def create(self, **_kwargs):
                FakeOpenAI.calls += 1
                raise RuntimeError("boom")

        class FakeOpenAI:
            init_kwargs = []
            calls = 0

            def __init__(self, **kwargs) -> None:
                FakeOpenAI.init_kwargs.append(kwargs)
                self.chat = SimpleNamespace(completions=FakeCompletions())

        reasoner = ConfigurationReasoner(
            LLMConfig(
                api_key="test-key",
                base_url="https://llm.example.com/v1",
                max_retries=3,
                retry_delay=0,
            )
        )
        prompt, _ = reasoner._create_prompts([], "Return JSON only.", ["k1"])

        with patch("llm.reasoner.OpenAI", FakeOpenAI):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                reasoner._execute_llm_call(
                    prompt,
                    "Query: select 1",
                    ["k1"],
                    return_format="vector",
                )

        self.assertEqual(FakeOpenAI.calls, 3)
        self.assertTrue(
            all(kwargs["max_retries"] == 0 for kwargs in FakeOpenAI.init_kwargs)
        )

    def test_remote_llm_calls_require_enabled_and_endpoint_or_key(self) -> None:
        self.assertFalse(LLMConfig(api_key="", base_url="").can_call_remote())
        self.assertFalse(LLMConfig(enabled=False, api_key="test-key").can_call_remote())
        self.assertTrue(LLMConfig(api_key="test-key").can_call_remote())
        self.assertTrue(LLMConfig(base_url="http://localhost:8000/v1").can_call_remote())
        self.assertTrue(LLMConfig(base_url="http://127.0.0.1:8000/v1").can_call_remote())
        self.assertFalse(LLMConfig(base_url="https://llm.example.com/v1").can_call_remote())
        self.assertTrue(
            LLMConfig(
                api_key="test-key",
                base_url="https://llm.example.com/v1",
            ).can_call_remote()
        )

    def test_default_reasoner_temperature_matches_paper(self) -> None:
        self.assertAlmostEqual(LLMConfig().temperature, 0.7)
        self.assertAlmostEqual(ConfigurationReasoner(LLMConfig())._synthesis_temperature(), 1.0)

    def test_disabled_reasoner_refuses_remote_call(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as sql_file:
            sql_file.write("select * from t")
            sql_path = sql_file.name

        try:
            reasoner = ConfigurationReasoner(
                LLMConfig(enabled=False, api_key="test-key", retry_delay=0)
            )
            with self.assertRaisesRegex(RuntimeError, "Remote LLM calls are disabled"):
                reasoner.recommend_next_configs(
                    observations=[([0.2], 1000.0)],
                    sql_path=sql_path,
                    knob_names=["k1"],
                    batch=1,
                )
        finally:
            Path(sql_path).unlink(missing_ok=True)

    def test_reasoner_requires_in_context_observations(self) -> None:
        reasoner = ConfigurationReasoner(LLMConfig(api_key="test-key"))
        with self.assertRaisesRegex(ValueError, "in-context observations"):
            reasoner.recommend_next_configs(
                observations=[],
                sql_path="unused.sql",
                knob_names=[
                    "tidb_opt_hash_join_cost_factor",
                    "tidb_join_order_cost_factor:title",
                ],
                batch=2,
            )

        with self.assertRaisesRegex(ValueError, "in-context observations"):
            reasoner.recommend_next_configs_after_rejection(
                observations=[],
                sql_path="unused.sql",
                last_failed_vectors=[[0.1, 0.2]],
                knob_names=[
                    "tidb_opt_hash_join_cost_factor",
                    "tidb_join_order_cost_factor:title",
                ],
                batch=2,
            )

    def test_prompt_uses_actual_c_knob_names(self) -> None:
        prompt = normal_config_prompt(
            1,
            "demo",
            [
                "tidb_opt_hash_join_cost_factor",
                "tidb_join_order_cost_factor:title",
            ],
        )

        self.assertIn("tidb_opt_hash_join_cost_factor", prompt)
        self.assertIn("tidb_join_order_cost_factor:title", prompt)
        self.assertNotIn('"knob_0"', prompt)

    def test_prompt_output_shape_matches_requested_batch(self) -> None:
        prompt = normal_config_prompt(
            3,
            "demo",
            ["k1", "k2"],
        )

        output_shape_text = prompt.split("Output shape:", 1)[1].strip()
        output_shape = json.loads(output_shape_text)

        self.assertEqual(len(output_shape), 3)
        self.assertTrue(all(set(config) == {"k1", "k2"} for config in output_shape))

    def test_knob_description_treats_llm_values_as_normalized_coordinates(self) -> None:
        reasoner = ConfigurationReasoner(LLMConfig(api_key="test-key"))

        description = reasoner._get_knob_descriptions(
            [
                "tidb_opt_hash_join_cost_factor",
                "tidb_join_order_cost_factor:title",
            ]
        )

        self.assertIn("normalized OBELISK coordinates", description)
        self.assertIn("log-space denormalization", description)
        self.assertIn("not physical cost factors", description)
        self.assertIn("tidb_opt_hash_join_cost_factor", description)
        self.assertIn("tidb_join_order_cost_factor:title", description)
        self.assertNotIn("0.5 means the original cost remains unchanged", description)

    def test_reasoner_includes_guider_xbo_in_llm_input(self) -> None:
        class CapturingReasoner(ConfigurationReasoner):
            def __init__(self) -> None:
                super().__init__(LLMConfig(api_key="test-key", retry_delay=0))
                self.input_info = ""

            def _execute_llm_call(self, prompt, input_info, knob_names, return_format="dict", temperature=None):
                self.input_info = input_info
                return [[0.4, 0.6]]

        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as sql_file:
            sql_file.write("select * from t where a > 1")
            sql_path = sql_file.name

        try:
            reasoner = CapturingReasoner()
            vectors = reasoner.recommend_next_configs(
                observations=[([0.2, 0.8], 1000.0)],
                sql_path=sql_path,
                knob_names=["k1", "k2"],
                batch=1,
                guider_vector=[0.25, 0.75],
            )
        finally:
            Path(sql_path).unlink(missing_ok=True)

        self.assertEqual(vectors, [[0.4, 0.6]])
        self.assertIn("Guider proposal xBO", reasoner.input_info)
        self.assertIn('"k1": 0.25', reasoner.input_info)
        self.assertIn('"k2": 0.75', reasoner.input_info)

    def test_json_extraction_accepts_object_or_array_in_text(self) -> None:
        reasoner = ConfigurationReasoner(LLMConfig(api_key="test-key"))

        self.assertEqual(reasoner._extract_json_config_batch('{"knob_0": 0.5}'), [{"knob_0": 0.5}])
        self.assertEqual(
            reasoner._extract_json_config_batch('```json\n[{"knob_0": 0.5}]\n```'),
            [{"knob_0": 0.5}],
        )

    def test_malformed_llm_values_default_without_breaking_workflow(self) -> None:
        reasoner = ConfigurationReasoner(LLMConfig(api_key="test-key"))

        vectors = reasoner._process_configs_to_vectors(
            [{"k1": "bad", "k2": 1.7, "k3": float("nan"), "k4": True}],
            ["k1", "k2", "k3", "k4"],
        )

        self.assertEqual(vectors, [[0.5, 0.5, 0.5, 0.5]])

    def test_legacy_llm_name_is_reasoner_alias(self) -> None:
        self.assertIs(LLM, ConfigurationReasoner)

    def test_rejection_aware_reasoner_uses_critique_then_synthesis(self) -> None:
        class CapturingReasoner(ConfigurationReasoner):
            def __init__(self) -> None:
                super().__init__(LLMConfig(api_key="test-key", retry_delay=0))
                self.critique_text = ""
                self.synthesis_text = ""
                self.critique_input = ""
                self.synthesis_input = ""
                self.critique_temperature = None
                self.synthesis_temperature = None

            def _execute_text_llm_call(self, prompt, input_info, temperature=None):
                self.critique_input = input_info
                self.critique_temperature = temperature
                self.critique_text = "\n".join(
                    str(message.content)
                    for message in prompt.invoke({"input": input_info}).to_messages()
                )
                return "Rejected configurations overuse high k1 and ignore k2 interactions."

            def _execute_llm_call(self, prompt, input_info, knob_names, return_format="dict", temperature=None):
                self.synthesis_input = input_info
                self.synthesis_temperature = temperature
                self.synthesis_text = "\n".join(
                    str(message.content)
                    for message in prompt.invoke({"input": input_info}).to_messages()
                )
                return [[0.123456, 0.876543]]

        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as sql_file:
            sql_file.write("select * from t where a > 1")
            sql_path = sql_file.name

        try:
            reasoner = CapturingReasoner()
            vectors = reasoner.recommend_next_configs_after_rejection(
                observations=[([0.2, 0.8], 1000.0), ([0.7, 0.3], 1200.0)],
                sql_path=sql_path,
                last_failed_vectors=[[0.95, 0.95]],
                knob_names=["k1", "k2"],
                batch=1,
                guider_vector=[0.4, 0.6],
            )
        finally:
            Path(sql_path).unlink(missing_ok=True)

        self.assertEqual(vectors, [[0.123456, 0.876543]])
        self.assertIn("previous batch", reasoner.critique_text)
        self.assertIn("rejected", reasoner.critique_text.lower())
        self.assertIn("Self-critique", reasoner.synthesis_text)
        self.assertIn("overuse high k1", reasoner.synthesis_text)
        self.assertIn("Guider proposal xBO", reasoner.critique_input)
        self.assertIn("Guider proposal xBO", reasoner.synthesis_input)
        self.assertAlmostEqual(reasoner.critique_temperature, 0.7)
        self.assertAlmostEqual(reasoner.synthesis_temperature, 1.0)
        self.assertGreater(reasoner.synthesis_temperature, reasoner.critique_temperature)

    def test_reasoner_passes_llm_limits_to_prompt_optimizer(self) -> None:
        class CapturingPromptOptimizer:
            init_kwargs = {}

            def __init__(self, **kwargs) -> None:
                CapturingPromptOptimizer.init_kwargs = kwargs

            def optimize_prompt(
                self,
                base_prompt,
                sql,
                observations,
                failed_configs,
                max_iterations,
                top_n,
            ):
                return base_prompt

        class CapturingReasoner(ConfigurationReasoner):
            def _execute_text_llm_call(self, prompt, input_info, temperature=None):
                return "Avoid rejected configurations."

            def _execute_llm_call(self, prompt, input_info, knob_names, return_format="dict", temperature=None):
                return [[0.123456]]

        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as sql_file:
            sql_file.write("select * from t where a > 1")
            sql_path = sql_file.name

        try:
            reasoner = CapturingReasoner(
                LLMConfig(
                    api_key="test-key",
                    base_url="https://llm.example.com/v1",
                    model_name="provider-chat",
                    temperature=0.2,
                    max_retries=4,
                    max_new_tokens=111,
                    top_p=0.6,
                    prompt_optimizer_enabled=True,
                    prompt_optimizer_iterations=2,
                    prompt_optimizer_top_n=3,
                )
            )
            with patch("llm.reasoner.PromptOptimizer", CapturingPromptOptimizer):
                vectors = reasoner.recommend_next_configs_after_rejection(
                    observations=[([0.2], 1000.0)],
                    sql_path=sql_path,
                    last_failed_vectors=[[0.8]],
                    knob_names=["k1"],
                    batch=1,
                    guider_vector=[0.3],
                )
        finally:
            Path(sql_path).unlink(missing_ok=True)

        self.assertEqual(vectors, [[0.123456]])
        self.assertEqual(CapturingPromptOptimizer.init_kwargs["model_name"], "provider-chat")
        self.assertEqual(CapturingPromptOptimizer.init_kwargs["api_key"], "test-key")
        self.assertEqual(CapturingPromptOptimizer.init_kwargs["base_url"], "https://llm.example.com/v1")
        self.assertAlmostEqual(CapturingPromptOptimizer.init_kwargs["temperature"], 0.5)
        self.assertEqual(CapturingPromptOptimizer.init_kwargs["max_retries"], 4)
        self.assertEqual(CapturingPromptOptimizer.init_kwargs["max_new_tokens"], 111)
        self.assertEqual(CapturingPromptOptimizer.init_kwargs["top_p"], 0.6)

    def test_prompt_optimization_preserves_final_output_contract(self) -> None:
        class LoosePromptOptimizer:
            def __init__(self, **_kwargs) -> None:
                pass

            def optimize_prompt(self, **_kwargs):
                return "Use database intuition to suggest a better configuration."

        class CapturingReasoner(ConfigurationReasoner):
            def __init__(self) -> None:
                super().__init__(
                    LLMConfig(
                        api_key="test-key",
                        prompt_optimizer_enabled=True,
                        retry_delay=0,
                    )
                )
                self.synthesis_text = ""

            def _execute_text_llm_call(self, prompt, input_info, temperature=None):
                return "Avoid the rejected high-high pattern."

            def _execute_llm_call(self, prompt, input_info, knob_names, return_format="dict", temperature=None):
                self.synthesis_text = "\n".join(
                    str(message.content)
                    for message in prompt.invoke({"input": input_info}).to_messages()
                )
                return [[0.123456, 0.654321]]

        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as sql_file:
            sql_file.write("select * from t where a > 1")
            sql_path = sql_file.name

        try:
            reasoner = CapturingReasoner()
            with patch("llm.reasoner.PromptOptimizer", LoosePromptOptimizer):
                vectors = reasoner.recommend_next_configs_after_rejection(
                    observations=[([0.2, 0.8], 1000.0)],
                    sql_path=sql_path,
                    last_failed_vectors=[[0.95, 0.95]],
                    knob_names=["k1", "k2"],
                    batch=1,
                    guider_vector=[0.3, 0.4],
                )
        finally:
            Path(sql_path).unlink(missing_ok=True)

        self.assertEqual(vectors, [[0.123456, 0.654321]])
        self.assertIn("Final output contract", reasoner.synthesis_text)
        self.assertIn("Return only a JSON array", reasoner.synthesis_text)
        self.assertIn("Output shape", reasoner.synthesis_text)
        self.assertIn('"k1"', reasoner.synthesis_text)
        self.assertIn('"k2"', reasoner.synthesis_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
