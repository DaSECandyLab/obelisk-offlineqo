import json
import yaml
import re
import random
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from util.logger import logger


class PromptPool:
    def __init__(self, prompt_pool_path: str):
        with open(prompt_pool_path, 'r') as f:
            self.pool = yaml.safe_load(f)

    def __getattr__(self, name):
        return self.pool[name]


def extract_between(start: str, end: str, text: str) -> str:
    """Extract text between two delimiters."""
    start_idx = text.find(start)
    if start_idx == -1:
        return ""
    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    return text[start_idx:end_idx] if end_idx != -1 else ""


class PromptOptimizer:
    def __init__(
        self,
        prompt_pool_path: str,
        model_name: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.7,
        max_retries: int = 5,
        max_new_tokens: int = 2048,
        top_p: float = 0.7,
    ):
        self.prompt_pool = PromptPool(prompt_pool_path)
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = top_p
        self.base_url = base_url
        self.openai_client = None
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_retries": max_retries,
            "top_p": top_p,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            self.openai_client = OpenAI(
                api_key=api_key or "EMPTY",
                base_url=base_url,
                max_retries=max_retries,
            )
            self.model = None
        else:
            kwargs["max_completion_tokens"] = self.max_new_tokens
            self.model = ChatOpenAI(**kwargs)

    def _chat_completion(self, user_prompt: str, system_prompt: str = None) -> str:
        """Wrapper for LLM chat completion."""
        system_prompt = system_prompt or self.prompt_pool.system_prompt

        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]

        if self.openai_client is not None:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
            )
            return response.choices[0].message.content or ""

        prompt_template = ChatPromptTemplate.from_messages(messages)
        chain = prompt_template | self.model
        return chain.invoke({}).content

    def _generate_different_styles(self, base_instruction: str, mutation_rounds: int = 2, style_count: int = 5) -> List[str]:
        """Generate different prompt variations by mixing thinking styles."""
        candidate_prompts = [base_instruction]

        for _ in range(mutation_rounds):
            # Select a subset of thinking styles
            selected_styles = random.sample(self.prompt_pool.thinking_styles, style_count)

            # Create the mutation prompt
            mutation_prompt = self.prompt_pool.meta_sample_template.format(
                meta_prompts="\n".join(selected_styles),
                num_variations=style_count,
                prompt_instruction=base_instruction
            )

            # Generate variations
            generated = self._chat_completion(mutation_prompt)
            variations = re.findall(r'<START>(.*?)<END>', generated, re.DOTALL)

            # Clean and add to candidates
            for var in variations:
                var = var.strip()
                if var and var not in candidate_prompts:
                    candidate_prompts.append(var)

        return candidate_prompts

    def _critique_and_refine(self, prompt: str, further_enhance: bool = False) -> str:
        """Generate critique and refine the prompt."""
        if further_enhance:
            critique_prompt = self.prompt_pool.meta_positive_critique_template.format(
                instruction=prompt
            )
        else:
            critique_prompt = self.prompt_pool.meta_critique_template.format(
                instruction=prompt
            )

        # Generate critique
        critique = self._chat_completion(critique_prompt, self.prompt_pool.expert_profile)

        # Refine the prompt
        refine_prompt = self.prompt_pool.critique_refine_template.format(
            instruction=prompt,
            critique=critique,
            steps_per_sample=1
        )

        refined = self._chat_completion(refine_prompt, self.prompt_pool.expert_profile)
        refined_prompts = re.findall(r'<START>(.*?)<END>', refined, re.DOTALL)

        if refined_prompts:
            return refined_prompts[0].strip()
        return prompt  # Fall back to original if refinement fails

    def _get_prompt_score(self, prompts: List[str], sql: str, observations: List[Tuple[Dict, float]],
                         failed_configs: List[Dict], batch: int = 1) -> List[Tuple[str, float]]:
        """Evaluate prompt performance by generating configurations."""
        prompt_scores = []

        for prompt in prompts:
            try:
                # Create examples string
                examples_str = "\n".join([
                    f"Configuration: {json.dumps(config)} | Execution Time: {score}"
                    for config, score in observations
                ])

                # Create failed configs string
                failed_str = "\n".join([json.dumps(config) for config in failed_configs])

                # Generate the solve prompt
                solve_prompt = self.prompt_pool.solve_template.format(
                    instruction=prompt,
                    sql=sql,
                    examples=examples_str,
                    failed_configs=failed_str,
                    batch=batch
                )

                # Generate configurations
                response = self._chat_completion(solve_prompt, self.prompt_pool.expert_profile)

                # Simple scoring: check if valid JSON array is generated
                if re.search(r'\[\s*{.*?}\s*\]', response, re.DOTALL):
                    # Give a base score if valid JSON is generated
                    prompt_scores.append((prompt, 1.0))
                else:
                    prompt_scores.append((prompt, 0.0))

            except Exception as e:
                logger.error(f"Error evaluating prompt '{prompt}': {e}")
                prompt_scores.append((prompt, 0.0))

        return prompt_scores

    def optimize_prompt(self, base_prompt: str, sql: str, observations: List[Tuple[Dict, float]],
                       failed_configs: List[Dict], max_iterations: int = 3, top_n: int = 5) -> str:
        """
        Optimize the base prompt for rejection-aware configuration generation.

        Args:
            base_prompt: The original system prompt
            sql: The SQL query to optimize
            observations: Historical observations (config, score)
            failed_configs: Failed configurations to avoid
            max_iterations: Number of optimization iterations
            top_n: Number of top prompts to keep

        Returns:
            The optimized prompt
        """
        current_prompts = [base_prompt]

        for iteration in range(max_iterations):
            logger.info(f"Prompt optimization iteration {iteration + 1}/{max_iterations}")

            # Generate variations
            logger.debug("Generating prompt variations...")
            variations = []
            for prompt in current_prompts:
                variations.extend(self._generate_different_styles(prompt))

            # Evaluate variations
            logger.debug("Evaluating prompt variations...")
            prompt_scores = self._get_prompt_score(variations, sql, observations, failed_configs)

            # Sort and select top n
            logger.debug("Selecting top prompts...")
            sorted_prompts = sorted(prompt_scores, key=lambda x: x[1], reverse=True)[:top_n]
            top_prompts = [prompt for prompt, _ in sorted_prompts]

            # Refine top prompts
            logger.debug("Refining top prompts...")
            refined_prompts = []
            for prompt in top_prompts:
                refined = self._critique_and_refine(prompt)
                refined_prompts.append(refined)

            current_prompts = refined_prompts

        # Select the best prompt from the final set
        if not current_prompts:
            return base_prompt

        return current_prompts[0]
