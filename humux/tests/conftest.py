"""Shared test fixtures and pytest configuration.

Fixtures defined here are available to all test files in this directory.
A test file may override any fixture by defining its own version — that
takes precedence over this module's definition.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (integration/performance)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "asyncio: async test running via pytest-asyncio")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> "AgentCore":  # noqa: F821
    """A bare AgentCore operating in tmp_path.

    Override this fixture in a test file when you need a different
    configuration (llm_provider, model, feature flags, etc.).
    """
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore
    from core.config import Config

    return AgentCore(Config())


@pytest.fixture
def configured_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> "AgentCore":  # noqa: F821
    """An AgentCore with common production-like defaults.

    - LLM provider: deepseek / deepseek-v4-flash
    - Embeddings disabled (avoids model load)
    - Task reflection & goal decomposition disabled

    Most concurrency / steering tests use this profile.
    """
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore
    from core.config import Config

    cfg = Config()
    cfg.agent.llm_provider = "deepseek"
    cfg.agent.model = "deepseek-v4-flash"
    cfg.memory.embedding.enabled = False
    cfg.task_reflection.enabled = False
    cfg.goal_decomposition.enabled = False
    return AgentCore(cfg)
