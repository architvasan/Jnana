"""Regression tests for structured output and concurrent persistence repairs."""
from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from typing import cast

from jnana.protognosis.core.agent_core import ContextMemory, ResearchHypothesis, SupervisorAgent
from jnana.protognosis.core.llm_interface import LLMInterface, OpenAILLM


class TuplePlanLLM:
    def generate_with_json_output(self, *_args, **_kwargs):
        return ({"main_objective": "goal", "domain": "biology", "evaluation_criteria": ["testability"]}, 1, 1)


class RetryingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        content = "" if self.calls == 1 else '{"ok": true}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )


def test_research_goal_parser_unwraps_provider_usage_tuple() -> None:
    memory = ContextMemory()
    supervisor = SupervisorAgent(cast(LLMInterface, TuplePlanLLM()), memory, max_workers=1)
    plan = supervisor.parse_research_goal("test goal")
    assert plan["main_objective"] == "goal"
    assert plan["original_research_goal"] == "test goal"


def test_openai_json_output_retries_empty_content() -> None:
    llm = object.__new__(OpenAILLM)
    llm.model = "test"
    llm.model_adapter = {"omit_temperature": True}
    llm.total_calls = llm.total_prompt_tokens = llm.total_completion_tokens = 0
    completions = RetryingCompletions()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result, prompt_tokens, completion_tokens = llm.generate_with_json_output("prompt", {"ok": "boolean"})
    assert result == {"ok": True}
    assert completions.calls == 2
    assert (prompt_tokens, completion_tokens) == (2, 3)


def test_concurrent_hypothesis_additions_are_not_lost() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        memory = ContextMemory(str(path))
        hypotheses = [ResearchHypothesis(str(i), str(i), "test", hypothesis_id=str(i)) for i in range(40)]
        threads = [threading.Thread(target=memory.add_hypothesis, args=(hypothesis,)) for hypothesis in hypotheses]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert len(memory.get_all_hypotheses()) == 40
        assert len(persisted["hypotheses"]) == 40
