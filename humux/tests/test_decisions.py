"""TypeSafe System One API contract and turn routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.agent import AgentCore
from core.config import Config, DecisionConfig
from core.decisions import DecisionClient
from core.models import AgentResponse
from core.permissions import PermissionLevel


def noul(probability):
    return {"type": "noul", "noul": probability}


def choice(value, probability):
    return {"type": "choice", "choice": value, "probabilities": {value: probability}}


@pytest.mark.asyncio
async def test_typesafe_api_contract(monkeypatch):
    def handler(request):
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        data = json.loads(request.content)
        assert data["model"] == "jev-test"
        assert data["state"]["latest_message"] == "please react"
        assert data["questions"]["reply"]["type"] == "noul"
        assert "none" in data["questions"]["reaction"]["criteria"]
        return httpx.Response(
            200,
            json={
                "answers": {
                    "reply": noul(0.93),
                    "reaction": choice("thumbsup", 0.98),
                    "reaction_only": noul(0.96),
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("core.decisions.httpx.AsyncClient", lambda **kwargs: client)
    decider = DecisionClient(DecisionConfig(model="jev-test", api_key="test-key"))
    result = await decider.incoming(
        "please react", "Assistant", history=[], others=[], gate_reply=True, can_react=True
    )
    assert result.reply and result.reaction == "thumbsup" and result.reaction_only


@pytest.mark.parametrize("response", [httpx.Response(429), httpx.Response(200, json=[])])
@pytest.mark.asyncio
async def test_provider_failure_defaults_to_reply_and_text(monkeypatch, response):
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "core.decisions.httpx.AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(lambda _request: response)),
    )
    decider = DecisionClient(DecisionConfig(api_key="test-key"))
    decision = await decider.incoming(
        "hello", "Assistant", history=[], others=[], gate_reply=True, can_react=True
    )
    assert decision.reply and decision.reaction is None
    assert await decider.spoken_parts("hello", ["hi"]) == set()


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.decisions.provider = "typesafe"
    cfg.voice.tts_enabled = True
    cfg.reply_decision.enabled = True
    cfg.memory.embedding.enabled = False
    cfg.goal_decomposition.enabled = False
    agent = AgentCore(cfg)
    agent.channels["telegram"] = SimpleNamespace(react=AsyncMock())
    return agent


@pytest.mark.asyncio
async def test_reply_gate_and_reaction_use_batched_decision(agent, monkeypatch):
    evaluate = AsyncMock(
        return_value={
            "reply": noul(0.06),
            "reaction": choice("heart", 0.99),
            "reaction_only": noul(0.99),
        }
    )
    monkeypatch.setattr(DecisionClient, "_evaluate", evaluate)
    generate = AsyncMock(return_value=AgentResponse(text="Generated"))
    monkeypatch.setattr(agent, "_process_injection", generate)
    monkeypatch.setattr(agent, "_process_session", generate)

    result = await agent.process(
        "A message for another bot",
        "telegram",
        "111",
        chat_id="-111",
        addressed=False,
        message_id=32,
    )
    assert result.text == ""
    generate.assert_not_awaited()
    agent.channels["telegram"].react.assert_not_awaited()
    assert set(evaluate.call_args.args[1]) == {"reply", "reaction", "reaction_only"}
    assert not agent._reply_times.get(("telegram", "-111"))


@pytest.mark.asyncio
async def test_reaction_only_respects_permissions(agent, monkeypatch):
    evaluate = AsyncMock(
        return_value={
            "reaction": choice("thumbsup", 0.99),
            "reaction_only": noul(0.99),
        }
    )
    monkeypatch.setattr(DecisionClient, "_evaluate", evaluate)
    generate = AsyncMock(return_value=AgentResponse(text="Generated"))
    monkeypatch.setattr(agent, "_process_injection", generate)
    monkeypatch.setattr(agent, "_process_session", generate)

    result = await agent.process(
        "Only react with a thumbs up", "telegram", "111", chat_id="111", message_id=32
    )
    assert result.text == ""
    generate.assert_not_awaited()
    agent.channels["telegram"].react.assert_awaited_once_with("111", 32, "thumbsup")
    assert "reply" not in evaluate.call_args.args[1]
    assert "set_reaction" not in {tool["name"] for tool in agent._tools_for_turn(None)}

    agent.permissions.add_rule("set_reaction", PermissionLevel.NEVER)
    result = await agent.process(
        "Only react with a thumbs up", "telegram", "111", chat_id="111", message_id=33
    )
    assert result.text == "Generated"
    assert agent.channels["telegram"].react.await_count == 1


@pytest.mark.asyncio
async def test_voice_decision_overrides_generation_marker(agent, monkeypatch):
    evaluate = AsyncMock(return_value={"part_0": noul(0.03), "part_1": noul(0.93)})
    monkeypatch.setattr(DecisionClient, "_evaluate", evaluate)
    agent.voice = SimpleNamespace(synthesize=AsyncMock(return_value=b"speech"))

    messages, combined = await agent._split_reply(
        "A link [respond_with_voice:en][[split]]Ciao [respond_with_voice:it]",
        None,
        message="[voice] Hello",
        channel="telegram",
    )
    assert [m.voice for m in messages] == [None, b"speech"]
    assert combined == "A link\nCiao"
    agent.voice.synthesize.assert_awaited_once_with("Ciao", voice=None, lang="it")
    evaluate.assert_awaited_once()
    assert evaluate.call_args.args[0]["reply_parts"] == ["A link", "Ciao"]

    messages, _ = await agent._split_reply("Hello", None, channel="web")
    assert messages[0].voice is None
    assert evaluate.await_count == 1
