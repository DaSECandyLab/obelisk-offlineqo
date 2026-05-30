from __future__ import annotations

import json
from textwrap import dedent


def escape_chat_template(text: str) -> str:
    """Escape braces before passing static prompt text to ChatPromptTemplate."""
    return text.replace("{", "{{").replace("}", "}}")


def _format_knob_names(knob_names: list[str]) -> str:
    return json.dumps(knob_names, ensure_ascii=False)


def _format_output_shape(knob_names: list[str], batch: int) -> str:
    if not knob_names:
        knob_names = ["knob_0", "knob_1"]
    examples = []
    for example_idx in range(max(0, int(batch))):
        base = 0.123456 + (example_idx * 0.173219)
        examples.append(
            {
                name: round((base + idx * 0.111111) % 1.0, 6)
                for idx, name in enumerate(knob_names)
            }
        )
    return json.dumps(examples, indent=2)


def configuration_output_contract(batch: int, knob_names: list[str]) -> str:
    return dedent(
        f"""
        Final output contract:
        - Return only a JSON array. Do not include Markdown or explanation.
        - Generate exactly {batch} configurations.
        - Every value must be between 0.0 and 1.0.
        - Every configuration must contain every required knob exactly once.
        - Use precise decimal values, not only 0.5 or round tenths.

        Required knob names:
        {_format_knob_names(knob_names)}

        Output shape:
        {_format_output_shape(knob_names, batch)}
        """
    ).strip()


def ensure_configuration_output_contract(prompt: str, batch: int, knob_names: list[str]) -> str:
    """Append a non-optional output contract after any optimized prompt text."""
    return "\n\n".join(
        [
            prompt.strip(),
            "The following contract overrides any conflicting output-format text.",
            configuration_output_contract(batch, knob_names),
        ]
    )


def normal_config_prompt(batch: int, knob_description: str, knob_names: list[str]) -> str:
    return dedent(
        f"""
        You are tuning TiDB optimizer cost knobs for one SQL query.

        Goal:
        Recommend exactly {batch} new knob configurations that minimize
        execution time.

        Knob behavior:
        {knob_description}

        Use the historical configurations and execution times as evidence:
        - lower execution time is better;
        - use the Guider proposal xBO in the user message as the statistical
          center of the next search step;
        - exploit promising regions;
        - explore adjacent or under-tested regions when evidence is weak.

        {configuration_output_contract(batch, knob_names)}
        """
    ).strip()


def critique_prompt(
    failed_configs: str,
    knob_description: str,
    knob_names: list[str],
) -> str:
    return dedent(
        f"""
        You are the OBELISK Configuration Reasoner.

        The previous batch of TiDB C-knob configurations was rejected by the
        Evaluator admission gate. Analyze why the rejected configurations are
        likely flawed using the successful in-context examples and the SQL
        query supplied by the user message.

        Return concise structured feedback only. Do not propose new
        configurations yet.

        Knob behavior:
        {knob_description}

        Required knob names:
        {_format_knob_names(knob_names)}

        Rejected configurations:
        {failed_configs or "[]"}

        Feedback checklist:
        - Identify shared patterns in the rejected configurations.
        - Compare rejected configurations with the Guider proposal xBO if it is
          supplied in the user message.
        - Mention likely missing knob interactions.
        - Suggest what the next synthesis attempt should explicitly avoid.
        """
    ).strip()


def synthesis_prompt(
    batch: int,
    knob_description: str,
    critique: str,
    failed_configs: str,
    knob_names: list[str],
) -> str:
    return dedent(
        f"""
        You are the OBELISK Configuration Reasoner in synthesis mode.

        Your previous recommendations failed the Evaluator admission gate. Use
        the self-critique below as prompt optimization feedback, then generate
        a new batch of candidate configurations that avoids the flawed patterns
        and explores different execution-plan regions around the Guider
        proposal xBO in the user message.

        Self-critique:
        {critique}

        Knob behavior:
        {knob_description}

        Failed configurations to avoid:
        {failed_configs or "[]"}

        Additional guidance:
        - Avoid failed configurations and near-duplicates.
        - Use precise decimal values with varied high/low interactions.

        {configuration_output_contract(batch, knob_names)}
        """
    ).strip()
