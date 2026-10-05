"""Paused team turns retain their model without changing agent defaults."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from octop_harness.agent import HarnessAgent
from octop_harness.config import HarnessAgentConfig
from octop_harness.manager import HarnessAgentManager
from octop_harness.middleware.turn_model import resolve_turn_model_ref
from octop_harness.registry import AgentEntry
from octop_harness.request import ChatRequest
from octop_harness.teams.tools import build_team_tools

from octop.config import OctopConfig
from octop.infra.agents.manager import AgentManager
from octop.infra.agents.teams.team_manager import wire_host_dispatch
from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.services import build_shared_services
from octop.infra.utils.paths import PathLayout

KEY = "octop_team_model_override"


class _State(TypedDict):
    messages: list[Any]


@pytest.fixture
async def runtime(tmp_path: Path):
    paths = PathLayout(tmp_path / "octop")
    paths.ensure_root()
    db = SqlitePool(paths.db)
    run_migrations(db)
    services = build_shared_services(db=db, paths=paths, config=OctopConfig())
    manager = AgentManager(repos=services.repos, paths=services.paths)
    services.repos.user_repo.create(username="resume", password_hash="h", role="user")
    services.repos.agent_repo.create(agent_id="host", user_id=1, name="Host", kind="team")
    harness = HarnessAgentManager(log_dir=tmp_path / "logs")
    agent = object.__new__(HarnessAgent)
    agent._config = HarnessAgentConfig(default_model="p/default")
    agent._protocols = {}
    agent._lifecycle_lock = threading.RLock()
    agent._in_flight = 0
    agent._close_pending = False
    for agent_id, instance in (("host", agent), ("expert", MagicMock())):
        harness._registry.add(
            AgentEntry(
                agent_id=agent_id,
                agent=instance,
                config=HarnessAgentConfig(),
                metadata={"user_id": 1, "name": agent_id},
                tags=[],
                created_at=datetime.now(UTC),
            )
        )
    team = harness.team
    team.set_processor(
        SimpleNamespace(on_reply=AsyncMock(), compose_followup=MagicMock(return_value="summary"))
    )
    release_peer = asyncio.Event()

    async def stream_peer(*_args: Any, **_kwargs: Any) -> dict:
        await release_peer.wait()
        return {"messages": []}

    wire_host_dispatch(team, is_team=lambda aid: aid == "host", stream_peer=stream_peer)
    ask = next(tool for tool in build_team_tools(team) if tool.name == "ask_agent")
    observed: dict[str, dict[str, Any]] = {}

    async def gate(_state: _State) -> dict:
        interrupt({"action_requests": [{"name": "execute", "args": {"command": "example"}}]})
        return {}

    async def after(state: _State) -> dict:
        cfg = get_config()["configurable"]
        observed[cfg["thread_id"]] = {
            "config": dict(cfg),
            "model": resolve_turn_model_ref(
                pick_default_ref=lambda: agent.config.default_model,
                configurable=cfg,
                messages=[],
                state=state,
            ),
            "peer": json.loads(await ask.ainvoke({"expert": "expert", "message": "research"})),
            "in_flight": agent._in_flight,
        }
        return {}

    builder = StateGraph(_State)
    builder.add_node("gate", gate)
    builder.add_node("after", after)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", "after")
    builder.add_edge("after", END)
    agent._graph = builder.compile(checkpointer=MemorySaver())
    manager._harness_manager = harness
    try:
        yield SimpleNamespace(
            manager=manager, harness=harness, agent=agent, team=team, observed=observed
        )
    finally:
        assert team.inbox is not None
        release_peer.set()
        team.inbox.cancel_worker()
        await asyncio.sleep(0)
        db.close()


async def _pause(
    runtime: Any,
    *,
    room: str = "room",
    model: str | None = "p/model-a",
    override: Any = ...,
    metadata: dict[str, Any] | None = None,
) -> None:
    if runtime.manager._repos.thread_repo.get(room) is None:
        runtime.manager._repos.thread_repo.insert(
            thread_id=room,
            agent_id="host",
            user_id=1,
            channel_type="dashboard",
            session_key="sk",
            last_active=0,
        )
    configurable = {"session_key": "sk", "peer_invoke_mode": "async"}
    if override is ...:
        override = model
    if override is not None:
        configurable[KEY] = override
    request = ChatRequest(
        messages=[],
        thread_id=room,
        user="1",
        agent_id="host",
        model=model,
        configurable=configurable,
    )
    config = request.to_runnable_config()
    if metadata:
        config["metadata"] = metadata
    result = await runtime.agent.graph.ainvoke({"messages": []}, config=config)
    assert result["__interrupt__"]


async def _resume(runtime: Any, room: str = "room") -> list[Any]:
    return [
        chunk async for chunk in runtime.manager.resume_hitl("host", room, [{"type": "approve"}])
    ]


async def test_resume_keeps_paused_model_and_real_ask_agent_job(runtime: Any) -> None:
    await _pause(runtime)
    runtime.manager._repos.thread_repo.update_composer("room", model_ref="p/model-b")
    assert runtime.manager._repos.thread_repo.get("room").model_ref == "p/model-b"
    runtime.agent.config.default_model = "p/model-b"

    await _resume(runtime)

    observed = runtime.observed["room"]
    assert observed["model"] == "p/model-a"
    assert observed["config"][KEY] == "p/model-a"
    assert observed["config"]["agent_id"] == "host"
    assert observed["config"]["user"] == "1"
    assert observed["peer"]["mode"] == "background"
    job = runtime.team.inbox.get(observed["peer"]["job_id"])
    assert job.metadata[KEY] == "p/model-a"
    assert (job.source_agent_id, job.source_thread_id, str(job.user_id)) == ("host", "room", "1")
    assert observed["in_flight"] == 1
    assert runtime.agent._in_flight == 0
    assert runtime.harness._cancel_events == {}
    assert runtime.agent.config.default_model == "p/model-b"


async def test_resume_rejects_checkpoint_user_of_another_thread_owner(runtime: Any) -> None:
    await _pause(runtime, metadata={"user": "2"})
    runtime.agent.config.default_model = "p/model-b"

    await _resume(runtime)

    assert runtime.observed["room"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["room"]["config"]


@pytest.mark.parametrize("override", [None, "", "p/other-model"])
async def test_resume_without_valid_override_uses_original_protocol(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, override: Any
) -> None:
    await _pause(runtime, override=override)
    runtime.agent.config.default_model = "p/model-b"
    original = runtime.harness.resume_hitl
    calls: list[dict[str, Any]] = []

    async def resume(*args: Any, **kwargs: Any):
        calls.append(kwargs)
        async for chunk in original(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(runtime.harness, "resume_hitl", resume)
    await _resume(runtime)

    assert calls == [{}]
    assert runtime.observed["room"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["room"]["config"]


async def test_resume_auto_model_uses_original_default(runtime: Any) -> None:
    await _pause(runtime, model=None)
    runtime.agent.config.default_model = "p/model-b"

    await _resume(runtime)

    assert runtime.observed["room"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["room"]["config"]


@pytest.mark.parametrize("metadata", [{"agent_id": "other"}, {"user": True}])
async def test_resume_rejects_invalid_checkpoint_identity(runtime: Any, metadata: dict) -> None:
    await _pause(runtime, metadata=metadata)
    runtime.agent.config.default_model = "p/model-b"

    await _resume(runtime)

    assert runtime.observed["room"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["room"]["config"]


@pytest.mark.parametrize("identity", ["missing-thread", "wrong-agent", "wrong-owner"])
async def test_resume_requires_active_team_thread(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    await _pause(runtime)
    runtime.agent.config.default_model = "p/model-b"
    thread_repo = runtime.manager._repos.thread_repo
    thread = thread_repo.get("room")
    replacement = {
        "missing-thread": None,
        "wrong-agent": replace(thread, agent_id="other"),
        "wrong-owner": replace(thread, user_id=2),
    }[identity]
    monkeypatch.setattr(thread_repo, "get", lambda _tid: replacement)

    await _resume(runtime)

    assert runtime.observed["room"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["room"]["config"]


@pytest.mark.parametrize("override", ["p/model-a", None])
async def test_peer_resume_restores_only_explicit_override(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    await _pause(runtime, override=override)
    row = runtime.manager.get_row("host")
    monkeypatch.setattr(runtime.manager, "get_row", lambda _aid: replace(row, kind="agent"))
    runtime.agent.config.default_model = "p/model-b"

    await _resume(runtime)

    cfg = runtime.observed["room"]["config"]
    assert runtime.observed["room"]["model"] == (override or "p/model-b")
    if override:
        assert cfg[KEY] == override
    else:
        assert KEY not in cfg


async def test_resume_does_not_copy_unknown_or_permission_metadata(runtime: Any) -> None:
    unknown = {"hitl_policy": "allow_all", "custom_permission": "admin", "unknown": "value"}
    await _pause(runtime, metadata=unknown)

    await _resume(runtime)

    cfg = runtime.observed["room"]["config"]
    assert cfg[KEY] == "p/model-a"
    assert cfg["session_key"] == "sk"
    assert cfg["peer_invoke_mode"] == "async"
    assert not unknown.keys() & cfg.keys()


async def test_resume_rejects_snapshot_of_another_thread(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _pause(runtime)
    runtime.agent.config.default_model = "p/model-b"
    original = runtime.agent.graph.aget_state

    async def state(config: dict):
        snapshot = await original(config)
        return snapshot._replace(
            config={"configurable": {**snapshot.config["configurable"], "thread_id": "other"}}
        )

    monkeypatch.setattr(runtime.agent.graph, "aget_state", state)
    await _resume(runtime)

    assert runtime.observed["room"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["room"]["config"]


async def test_resume_does_not_replay_changed_checkpoint(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _pause(runtime)
    graph = runtime.agent.graph
    original = graph.aget_state
    first_checkpoint = (await original({"configurable": {"thread_id": "room"}})).config
    reads = 0

    async def state(config: dict):
        nonlocal reads
        snapshot = await original(config)
        reads += 1
        if reads == 1:
            await graph.aupdate_state(config, {"messages": []})
        return snapshot

    monkeypatch.setattr(graph, "aget_state", state)
    with pytest.raises(ValueError, match="changed before HITL resume"):
        await _resume(runtime)

    assert runtime.observed == {}
    current = await original({"configurable": {"thread_id": "room"}})
    assert (
        current.config["configurable"]["checkpoint_id"]
        != first_checkpoint["configurable"]["checkpoint_id"]
    )
    assert runtime.agent._in_flight == 0
    assert runtime.harness._cancel_events == {}
    await _pause(runtime, room="default", override=None)
    runtime.agent.config.default_model = "p/model-b"
    await _resume(runtime, "default")
    assert runtime.observed["default"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["default"]["config"]


async def test_cached_resume_protocol_isolates_concurrent_threads(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _pause(runtime, room="a", model="p/model-a")
    await _pause(runtime, room="c", model="p/model-c")
    runtime.agent.config.default_model = "p/model-b"
    original = runtime.agent.graph.aget_state
    arrived = asyncio.Event()
    reads: dict[str, int] = {}
    waiting = 0

    async def state(config: dict):
        nonlocal waiting
        room = config["configurable"]["thread_id"]
        reads[room] = reads.get(room, 0) + 1
        if reads[room] == 2:
            waiting += 1
            if waiting == 2:
                arrived.set()
            await asyncio.wait_for(arrived.wait(), timeout=5)
        return await original(config)

    monkeypatch.setattr(runtime.agent.graph, "aget_state", state)
    await asyncio.gather(_resume(runtime, "a"), _resume(runtime, "c"))

    assert len(runtime.agent._protocols) == 1
    for room, model in (("a", "p/model-a"), ("c", "p/model-c")):
        observed = runtime.observed[room]
        assert observed["model"] == model
        assert observed["config"][KEY] == model
        job = runtime.team.inbox.get(observed["peer"]["job_id"])
        assert job.metadata[KEY] == model
        assert job.source_thread_id == room
    assert runtime.agent._in_flight == 0
    assert runtime.harness._cancel_events == {}
    assert runtime.agent.config.default_model == "p/model-b"


async def test_early_close_does_not_leak_model_into_next_resume(runtime: Any) -> None:
    await _pause(runtime)
    stream = runtime.manager.resume_hitl("host", "room", [{"type": "approve"}])
    await anext(stream)
    await stream.aclose()
    await _pause(runtime, room="default", override=None)
    runtime.agent.config.default_model = "p/model-b"

    await _resume(runtime, "default")

    assert runtime.observed["default"]["model"] == "p/model-b"
    assert KEY not in runtime.observed["default"]["config"]
