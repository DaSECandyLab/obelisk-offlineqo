import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

from llm.llm_config import LLMConfig
from llm.prompts import (
    critique_prompt,
    escape_chat_template,
    ensure_configuration_output_contract,
    normal_config_prompt,
    synthesis_prompt,
)
from llm.promptopt.prompt_optimizer import PromptOptimizer
from util.logger import logger


class ConfigurationReasoner:
    """LLM-based configuration Reasoner from OBELISK Section 7."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_app_config()

    def _create_model(self, temperature: Optional[float] = None) -> ChatOpenAI:
        """Create ChatOpenAI instance."""
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_retries": 0,
            "top_p": self.config.top_p,
        }
        if not self.config.base_url:
            kwargs["max_completion_tokens"] = self.config.max_new_tokens
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return ChatOpenAI(**kwargs)

    def _synthesis_temperature(self) -> float:
        """Use the paper's higher-temperature synthesis without a separate mode."""
        return min(1.0, max(self.config.temperature, self.config.temperature + 0.3))

    def _ensure_remote_call_allowed(self) -> None:
        if self.config.can_call_remote():
            return
        raise RuntimeError(
            "Remote LLM calls are disabled or missing credentials for a non-local endpoint"
        )

    def _load_sql(self, sql_path: str) -> str:
        """Load SQL file content."""
        with open(sql_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _get_knob_descriptions(self, knob_names: List[str]) -> str:
        """Generate knob descriptions from knob names."""
        descriptions = []

        join_order_knobs = [
            name for name in knob_names
            if name.startswith("tidb_join_order_cost_factor:")
        ]
        operator_knobs = [
            name for name in knob_names
            if not name.startswith("tidb_join_order_cost_factor:")
        ]

        if join_order_knobs:
            descriptions.append(
                "Logical C-knobs with prefix tidb_join_order_cost_factor: scale "
                "the join-order cost/cardinality factor for a specific base table; "
                "lower values make joining that table earlier more attractive."
            )
        
        if operator_knobs:
            descriptions.append(
                "Physical C-knobs scale the estimated cost of physical operators; "
                "higher values penalize that operator family and lower values "
                "make it more attractive while still allowing the optimizer to choose it."
            )

        if knob_names:
            descriptions.append("Tune exactly these knobs: " + ", ".join(knob_names))

        descriptions.append(
            "All values you output are normalized OBELISK coordinates in [0, 1], "
            "not physical cost factors. During evaluation, each coordinate is "
            "converted back to its physical C-knob value with the paper's "
            "log-space denormalization. Lower normalized values usually map to "
            "lower physical cost scaling, and higher normalized values usually "
            "map to higher physical cost scaling. Do not output physical values "
            "such as 1.0, 10.0, or 0.1 unless they are valid normalized "
            "coordinates in [0, 1]."
        )

        return "\n".join(descriptions)

    def _create_prompts(
        self,
        examples: List[Dict],
        system_prompt: str,
        knob_names: List[str],
    ) -> Tuple[ChatPromptTemplate, FewShotChatMessagePromptTemplate]:
        """Create prompt template."""
        example_prompt = ChatPromptTemplate.from_messages([
            ("ai", "Configuration: {configuration}"),
            ("human", "Execution time: {score}"),
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}")
        ])

        logger.debug("LLM prompt template created.")
        return final_prompt, few_shot_prompt

    def _execute_llm_call(
        self,
        prompt: ChatPromptTemplate,
        input_info: str,
        knob_names: List[str],
        return_format: str = "dict",
        temperature: Optional[float] = None,
    ) -> Union[List[Dict], List[List[float]]]:
        """Execute LLM call and return results in specified format."""
        self._ensure_remote_call_allowed()
        parser = StrOutputParser()

        for retry in range(self.config.max_retries):
            try:
                if self.config.base_url:
                    results = self._invoke_openai_compatible(
                        prompt,
                        input_info,
                        temperature=temperature,
                    )
                else:
                    model = self._create_model(temperature)
                    chain = prompt | model
                    chain_result = chain.invoke({"input": input_info})
                    logger.debug(f"LLM response: {chain_result}")
                    results = parser.invoke(chain_result)

                # Extract raw JSON configuration
                raw_configs = self._extract_json_config_batch(results)
                if not raw_configs:
                    raise ValueError("Failed to extract valid configuration from LLM response")
                if return_format == "vector":
                    return self._process_configs_to_vectors(raw_configs, knob_names)
                else:
                    return self._process_configs_to_dicts(raw_configs, knob_names)

            except Exception as e:
                logger.warning(f"Attempt {retry + 1}/{self.config.max_retries} failed: {str(e)}")
                if retry < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
                    raise

    def _execute_text_llm_call(
        self,
        prompt: ChatPromptTemplate,
        input_info: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Execute an LLM call whose output is free-form text."""
        self._ensure_remote_call_allowed()
        parser = StrOutputParser()

        for retry in range(self.config.max_retries):
            try:
                if self.config.base_url:
                    return self._invoke_openai_compatible(
                        prompt,
                        input_info,
                        temperature=temperature,
                    ).strip()

                model = self._create_model(temperature)
                chain = prompt | model
                chain_result = chain.invoke({"input": input_info})
                return parser.invoke(chain_result).strip()
            except Exception as e:
                logger.warning(f"Attempt {retry + 1}/{self.config.max_retries} failed: {str(e)}")
                if retry < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"All {self.config.max_retries} text attempts failed")
                    raise

    def _invoke_openai_compatible(
        self,
        prompt: ChatPromptTemplate,
        input_info: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Invoke an OpenAI-compatible chat completion endpoint directly."""
        prompt_value = prompt.invoke({"input": input_info})
        messages = [
            {
                "role": self._message_role(message.type),
                "content": str(message.content),
            }
            for message in prompt_value.to_messages()
        ]
        client = OpenAI(
            api_key=self.config.api_key or "EMPTY",
            base_url=self.config.base_url,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("OpenAI-compatible LLM response: %s", content)
        return content

    @staticmethod
    def _message_role(message_type: str) -> str:
        if message_type == "human":
            return "user"
        if message_type == "ai":
            return "assistant"
        if message_type == "system":
            return "system"
        return "user"

    def _process_configs_to_dicts(self, raw_configs: List[Dict], knob_names: List[str]) -> List[Dict]:
        """Process raw configuration as dictionary format, perform value constraints and missing parameter completion."""
        result = []
        for config in raw_configs:
            processed_config = {}
            for knob_name in knob_names:
                processed_config[knob_name] = self._coerce_knob_value(
                    config,
                    knob_name,
                )
            result.append(processed_config)
        return result
    
    def _process_configs_to_vectors(self, raw_configs: List[Dict], knob_names: List[str]) -> List[List[float]]:
        """Process raw configuration as vector format, perform value constraints and missing parameter completion."""
        result = []
        for config in raw_configs:
            vector = []
            for knob_name in knob_names:
                vector.append(self._coerce_knob_value(config, knob_name))
            result.append(vector)
        return result

    def _coerce_knob_value(self, config: Dict, knob_name: str) -> float:
        """Default missing, malformed, non-finite, or out-of-range values."""
        if knob_name not in config:
            logger.warning(f"Missing parameter {knob_name}, using default 0.5")
            return 0.5

        if isinstance(config[knob_name], bool):
            logger.warning(f"Invalid parameter {knob_name}, using default 0.5")
            return 0.5

        try:
            value = float(config[knob_name])
        except (TypeError, ValueError):
            logger.warning(f"Invalid parameter {knob_name}, using default 0.5")
            return 0.5

        if not math.isfinite(value):
            logger.warning(f"Non-finite parameter {knob_name}, using default 0.5")
            return 0.5

        if value < 0.0 or value > 1.0:
            logger.warning(f"Out-of-range parameter {knob_name}, using default 0.5")
            return 0.5

        return value

    def _extract_json_config_batch(self, text: str) -> Optional[List[Dict]]:
        """Extract batch JSON configurations from text."""
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed

        return None

    def _create_examples_from_vectors(self, observations: List[Tuple[List[float], float]], knob_names: List[str]) -> List[Dict]:
        """Create examples directly from vector observations."""
        examples = []
        for vector, score in observations:
            config_dict = {knob_names[i]: vector[i] for i in range(len(vector))}
            examples.append({"configuration": json.dumps(config_dict), "score": score})
        return examples

    def _failed_configs_text(self, failed_vectors: List[List[float]], knob_names: List[str]) -> str:
        failed_configs = []
        for vector in failed_vectors:
            failed_configs.append({
                knob_names[i]: vector[i]
                for i in range(min(len(vector), len(knob_names)))
            })
        return "\n".join(json.dumps(config) for config in failed_configs)

    def _vector_to_named_config(
        self,
        vector: Optional[List[float]],
        knob_names: List[str],
    ) -> Dict[str, float]:
        if vector is None:
            return {}
        return {
            knob_names[i]: self._coerce_knob_value(
                {knob_names[i]: vector[i]},
                knob_names[i],
            )
            for i in range(min(len(vector), len(knob_names)))
        }

    def _query_input(
        self,
        sql: str,
        knob_names: List[str],
        guider_vector: Optional[List[float]] = None,
    ) -> str:
        parts = [f"Query: {sql}"]
        if guider_vector is not None:
            x_bo_config = self._vector_to_named_config(guider_vector, knob_names)
            parts.append(
                "Guider proposal xBO: "
                + json.dumps(x_bo_config, ensure_ascii=False, sort_keys=True)
            )
        return "\n\n".join(parts)

    def recommend_next_configs(
        self,
        observations: List[Tuple[List[float], float]],
        sql_path: str,
        knob_names: Optional[List[str]] = None,
        batch: int = 1,
        guider_vector: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """Recommend next batch of vectors directly."""
        if knob_names:
            dimension = len(knob_names)
        elif observations:
            dimension = len(observations[0][0])
            knob_names = [f"knob_{i}" for i in range(dimension)]
        else:
            dimension = 0
            knob_names = []

        if not observations:
            raise ValueError(
                "Reasoner requires in-context observations K; use Guider LHS fallback instead"
            )

        sql = self._load_sql(sql_path)
        examples = self._create_examples_from_vectors(observations, knob_names)
        knob_desc = self._get_knob_descriptions(knob_names)

        system_prompt = escape_chat_template(normal_config_prompt(batch, knob_desc, knob_names))

        prompt, _ = self._create_prompts(examples, system_prompt, knob_names)
        input_info = self._query_input(sql, knob_names, guider_vector)

        config_list = self._execute_llm_call(
            prompt,
            input_info,
            knob_names,
            return_format="vector",
            temperature=self.config.temperature,
        )
        return config_list

    def critique_rejections(
        self,
        observations: List[Tuple[List[float], float]],
        sql_path: str,
        rejected_vectors: List[List[float]],
        knob_names: List[str],
        guider_vector: Optional[List[float]] = None,
    ) -> str:
        """Run the §7.2 critique phase for rejected configurations."""
        if not observations or not rejected_vectors:
            return ""

        sql = self._load_sql(sql_path)
        examples = self._create_examples_from_vectors(observations, knob_names)
        knob_desc = self._get_knob_descriptions(knob_names)
        failed_text = self._failed_configs_text(rejected_vectors, knob_names)
        system_prompt = escape_chat_template(
            critique_prompt(failed_text, knob_desc, knob_names)
        )
        prompt, _ = self._create_prompts(examples, system_prompt, knob_names)
        return self._execute_text_llm_call(
            prompt,
            self._query_input(sql, knob_names, guider_vector),
            temperature=self.config.temperature,
        )

    def recommend_next_configs_after_rejection(
        self,
        observations: List[Tuple[List[float], float]],
        sql_path: str,
        last_failed_vectors: Optional[List[List[float]]] = None,
        knob_names: Optional[List[str]] = None,
        batch: int = 1,
        guider_vector: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """Recommend next batch using the Section 7.2 prompt-optimization loop."""
        last_failed_vectors = last_failed_vectors or []
        if knob_names:
            dimension = len(knob_names)
        elif observations:
            dimension = len(observations[0][0])
            knob_names = [f"knob_{i}" for i in range(dimension)]
        else:
            dimension = len(last_failed_vectors[0]) if last_failed_vectors else 0
            knob_names = [f"knob_{i}" for i in range(dimension)]

        if not observations:
            raise ValueError(
                "Reasoner requires in-context observations K; use Guider LHS fallback instead"
            )

        sql = self._load_sql(sql_path)
        examples = self._create_examples_from_vectors(observations, knob_names)
        knob_desc = self._get_knob_descriptions(knob_names)
        last_failed_text = self._failed_configs_text(last_failed_vectors, knob_names)
        synthesis_temperature = self._synthesis_temperature()

        critique = ""
        if last_failed_vectors:
            critique = self.critique_rejections(
                observations,
                sql_path,
                last_failed_vectors,
                knob_names,
                guider_vector=guider_vector,
            )
        if not critique:
            critique = (
                "The previous batch was rejected by the Evaluator. Avoid the "
                "failed configurations and generate a more diverse batch around xBO."
            )

        random.shuffle(examples)
        base_system_prompt = synthesis_prompt(
            batch,
            knob_desc,
            critique,
            last_failed_text,
            knob_names,
        )

        optimized_prompt = base_system_prompt
        if self.config.prompt_optimizer_enabled:
            try:
                logger.info("Starting rejection-aware prompt optimization...")

                prompt_optimizer = PromptOptimizer(
                    prompt_pool_path=os.path.join(os.path.dirname(__file__), 'promptopt', 'prompt_pool.yaml'),
                    model_name=self.config.model_name,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    temperature=synthesis_temperature,
                    max_retries=self.config.max_retries,
                    max_new_tokens=self.config.max_new_tokens,
                    top_p=self.config.top_p,
                )

                obs_dict = [(dict(zip(knob_names, vec)), score) for vec, score in observations]
                failed_configs = [
                    {
                        knob_names[i]: vec[i]
                        for i in range(min(len(vec), len(knob_names)))
                    }
                    for vec in last_failed_vectors
                ]

                optimized_prompt = prompt_optimizer.optimize_prompt(
                    base_prompt=base_system_prompt,
                    sql=sql,
                    observations=obs_dict,
                    failed_configs=failed_configs,
                    max_iterations=self.config.prompt_optimizer_iterations,
                    top_n=self.config.prompt_optimizer_top_n
                )

                logger.info(f"Optimized prompt: {optimized_prompt[:500]}...")

            except Exception as e:
                logger.error(f"Prompt optimization failed: {e}. Falling back to original prompt.")
                optimized_prompt = base_system_prompt

        optimized_prompt = ensure_configuration_output_contract(
            optimized_prompt,
            batch,
            knob_names,
        )
        system_prompt = escape_chat_template(optimized_prompt)

        prompt, _ = self._create_prompts(examples, system_prompt, knob_names)
        input_info = self._query_input(sql, knob_names, guider_vector)

        config_list = self._execute_llm_call(
            prompt,
            input_info,
            knob_names,
            return_format="vector",
            temperature=synthesis_temperature,
        )
        return config_list


LLM = ConfigurationReasoner
