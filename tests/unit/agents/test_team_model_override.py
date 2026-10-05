"""Conversation model overrides travel with jobs, not shared agent defaults."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from octop_harness.config import HarnessAgentConfig
from octop_harness.registry import AgentEntry, AgentRegistry
from octop_harness.teams.team_manager import TeamManager as HarnessTeam
from octop_harness.teams.tools import build_team_tools

from octop.infra.agents.teams import team_manager as team_mod

KEY = "octop_team_model_override"


@pytest.mark.asyncio
async def test_real_ask_agent_tool_captures_and_applies_its_runnable_config() -> None:
    team = _team()
    requests = asyncio.Queue()
    release = asyncio.Event()

    async def stream(request: object, **_kwargs: object) -> dict:
        requests.put_nowait(request)
        await release.wait()
        return {"messages": []}

    team_mod.wire_host_dispatch(team, is_team=lambda aid: aid == "host", stream_peer=stream)
    tool = next(tool for tool in build_team_tools(team) if tool.name == "ask_agent")
    config = _config("p/selected")
    config["configurable"]["peer_invoke_mode"] = "async"
    try:
        reply = json.loads(
            await tool.ainvoke({"expert": "expert", "message": "research"}, config=config)
        )
        assert reply["mode"] == "background"
        request = await asyncio.wait_for(requests.get(), timeout=5)
        assert request.to_runnable_config()["configurable"]["model"] == "p/selected"
        assert request.agent_id == "expert"
        assert request.thread_id == "room~expert"
    finally:
        release.set()
        assert team.inbox is not None
        team.inbox.cancel_worker()


def _team() -> HarnessTeam:
    registry = AgentRegistry()
    for agent_id in ("host", "expert", "other"):
        registry.add(
            AgentEntry(
                agent_id=agent_id,
                agent=MagicMock(),
                config=HarnessAgentConfig(),
                metadata={"user_id": 1, "name": agent_id},
                tags=[],
                created_at=datetime.now(UTC),
            )
        )
    team = HarnessTeam(registry)
    team.set_processor(
        SimpleNamespace(on_reply=AsyncMock(), compose_followup=MagicMock(return_value="summary"))
    )
    return team


def _kwargs(room: str = "room") -> dict:
    return {
        "from_agent_id": "host",
        "to_agent_id": "expert",
        "user_id": 1,
        "message": "research",
        "source_thread_id": room,
    }


def _config(model: str | None, *, room: str = "room", agent: str = "host") -> dict:
    return {"configurable": {"agent_id": agent, "thread_id": room, "user": 1, KEY: model}}


@pytest.mark.asyncio
async def test_jobs_keep_their_models_after_the_composer_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    current = _config("p/model-a")
    monkeypatch.setattr(team_mod, "get_config", lambda: current)
    stream = AsyncMock(return_value={"messages": []})
    team_mod.wire_host_dispatch(team, is_team=lambda aid: aid == "host", stream_peer=stream)
    metadata = {"session_key": "sk"}
    first = team.submit_peer(**_kwargs(), metadata=metadata)
    current = _config("p/model-b", room="room-b")
    second = team.submit_peer(**_kwargs("room-b"))
    current = _config(None)
    third = team.submit_peer(**_kwargs())
    assert team.inbox is not None
    team.inbox.cancel_worker()
    assert metadata == {"session_key": "sk"}
    assert team.inbox.get(first.job_id).metadata[KEY] == "p/model-a"
    assert team.inbox.get(second.job_id).metadata[KEY] == "p/model-b"
    assert KEY not in team.inbox.get(third.job_id).metadata
    # Worker context may have changed; the job's captured model is authoritative.
    for result, room, expected in (
        (first, "room", "p/model-a"),
        (second, "room-b", "p/model-b"),
        (third, "room", None),
    ):
        request, _ = await team._invoke_peer(
            **_kwargs(room), source="inbox", source_session_key="sk", job_id=result.job_id
        )
        cfg = request.to_runnable_config()["configurable"]
        assert cfg.get("model") == expected
        assert cfg["agent_id"] == "expert"
        assert cfg["user"] == "1"
    assert all(entry.config.default_model is None for entry in team._registry.values())


@pytest.mark.asyncio
async def test_sync_nested_peer_inherits_only_its_current_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    current = _config("p/unified", room="room~expert", agent="expert")
    monkeypatch.setattr(team_mod, "get_config", lambda: current)
    team_mod.wire_host_dispatch(team, is_team=lambda aid: aid == "host")
    kwargs = {
        "from_agent_id": "expert",
        "to_agent_id": "other",
        "user_id": 1,
        "message": "check",
        "source": "ask_agent",
        "source_thread_id": "room~expert",
        "source_session_key": "sk",
    }
    request = await team._build_peer_request(**kwargs)
    assert request.to_runnable_config()["configurable"]["model"] == "p/unified"
    assert request.configurable[KEY] == "p/unified"
    current = _config(None, room="room~expert", agent="expert")
    request = await team._build_peer_request(**kwargs)
    assert KEY not in request.configurable
    assert "model" not in request.to_runnable_config()["configurable"]


@pytest.mark.asyncio
async def test_dispatch_does_not_capture_another_callers_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    monkeypatch.setattr(team_mod, "get_config", lambda: _config("p/foreign", room="another-room"))
    team_mod.wire_host_dispatch(team, is_team=lambda aid: aid == "host")
    result = team.submit_peer(**_kwargs(), metadata={KEY: "p/stale"})
    assert team.inbox is not None
    team.inbox.cancel_worker()
    assert KEY not in team.inbox.get(result.job_id).metadata


@pytest.mark.asyncio
async def test_wrapup_and_redispatch_use_the_same_job_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    current = _config("p/selected")
    monkeypatch.setattr(team_mod, "get_config", lambda: current)
    stream = AsyncMock(return_value={"messages": []})
    team_mod.wire_host_dispatch(
        team, is_team=lambda aid: aid == "host", stream_peer=stream, stream_host=stream
    )
    result = team.submit_peer(**_kwargs())
    assert team.inbox is not None
    team.inbox.cancel_worker()
    current = _config(None)
    job = team.inbox.get(result.job_id)
    await team.inbox._synthesize_reply(job, "findings", None)
    request = stream.await_args.args[0]
    assert request.to_runnable_config()["configurable"]["model"] == "p/selected"
    assert request.configurable[KEY] == "p/selected"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["p/selected", None, ""])
async def test_wrapup_fallback_keeps_job_snapshot(
    monkeypatch: pytest.MonkeyPatch, model: str | None
) -> None:
    team = _team()
    team._registry.get("host").config.default_model = "p/default"
    current = _config(model)
    monkeypatch.setattr(team_mod, "get_config", lambda: current)
    requests = []

    async def stream_host(request: object, **_kwargs: object) -> dict:
        requests.append(request)
        if len(requests) == 1:
            raise RuntimeError("transient wrap-up failure")
        return {"messages": []}

    team_mod.wire_host_dispatch(
        team,
        is_team=lambda aid: aid == "host",
        stream_peer=AsyncMock(return_value={"messages": []}),
        stream_host=stream_host,
    )
    result = team.submit_peer(**_kwargs(), metadata={"session_key": "sk"})
    assert team.inbox is not None
    team.inbox.cancel_worker()
    current = _config("p/later")
    job = team.inbox.get(result.job_id)
    await team.inbox._synthesize_reply(job, "findings", None)

    assert len(requests) == 2
    for request in requests:
        cfg = request.to_runnable_config()["configurable"]
        assert cfg.get("model") == (model or None)
        assert cfg.get(KEY) == (model or None)
        assert cfg["agent_id"] == "host"
        assert cfg["user"] == "1"
    assert requests[1].thread_id == "room"
    assert requests[1].configurable["session_key"] == "sk"
    assert team._registry.get("host").config.default_model == "p/default"


@pytest.mark.asyncio
async def test_concurrent_wrapup_fallbacks_keep_each_room_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    current = _config("p/a", room="room-a")
    monkeypatch.setattr(team_mod, "get_config", lambda: current)
    requests = {"room-a": [], "room-b": []}
    both_fallbacks = asyncio.Event()

    async def stream_host(request: object, *, room_thread_id: str, **_kwargs: object) -> dict:
        calls = requests[room_thread_id]
        calls.append(request.to_runnable_config()["configurable"])
        if len(calls) == 1:
            raise RuntimeError("transient wrap-up failure")
        if all(len(items) >= 2 for items in requests.values()):
            both_fallbacks.set()
        await asyncio.wait_for(both_fallbacks.wait(), timeout=5)
        return {"messages": []}

    team_mod.wire_host_dispatch(
        team,
        is_team=lambda aid: aid == "host",
        stream_peer=AsyncMock(return_value={"messages": []}),
        stream_host=stream_host,
    )
    first = team.submit_peer(**_kwargs("room-a"))
    current = _config("p/b", room="room-b")
    second = team.submit_peer(**_kwargs("room-b"))
    assert team.inbox is not None
    team.inbox.cancel_worker()
    current = _config(None)
    await asyncio.gather(
        team.inbox._synthesize_reply(team.inbox.get(first.job_id), "a", None),
        team.inbox._synthesize_reply(team.inbox.get(second.job_id), "b", None),
    )
    for room, model in (("room-a", "p/a"), ("room-b", "p/b")):
        assert len(requests[room]) == 2
        assert all(cfg.get("model") == model and cfg.get(KEY) == model for cfg in requests[room])
    assert all(entry.config.default_model is None for entry in team._registry.values())


@pytest.mark.asyncio
async def test_non_team_wrapup_keeps_the_original_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    original_call = AsyncMock(return_value={"messages": []})
    team._call_agent = original_call
    monkeypatch.setattr(team_mod, "get_config", lambda: _config("p/selected"))
    stream = AsyncMock(return_value={"messages": []})
    team_mod.wire_host_dispatch(team, is_team=lambda _aid: False, stream_host=stream)
    result = team.submit_peer(**_kwargs())
    assert team.inbox is not None
    team.inbox.cancel_worker()
    await team.inbox._synthesize_reply(team.inbox.get(result.job_id), "findings", None)
    request = original_call.await_args.args[1]
    cfg = request.to_runnable_config()["configurable"]
    assert cfg.get("model") is None
    assert KEY not in cfg
    stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_wrapup_fallback_does_not_leak_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    monkeypatch.setattr(team_mod, "get_config", lambda: _config("p/selected"))
    requests = []

    async def stream_host(request: object, **_kwargs: object) -> dict:
        requests.append(request)
        if len(requests) == 1:
            raise RuntimeError("transient wrap-up failure")
        if len(requests) == 2:
            raise asyncio.CancelledError
        return {"messages": []}

    team_mod.wire_host_dispatch(
        team,
        is_team=lambda aid: aid == "host",
        stream_peer=AsyncMock(return_value={"messages": []}),
        stream_host=stream_host,
    )
    result = team.submit_peer(**_kwargs())
    assert team.inbox is not None
    team.inbox.cancel_worker()
    assert (
        await team.inbox._synthesize_reply(team.inbox.get(result.job_id), "findings", None) is None
    )

    request = team_mod.build_one_shot_request(
        user_id=1, agent_id="host", text="later", source="inbox", thread_id="room"
    )
    await team._call_agent("host", request)
    cfg = requests[2].to_runnable_config()["configurable"]
    assert cfg.get("model") is None
    assert KEY not in cfg
