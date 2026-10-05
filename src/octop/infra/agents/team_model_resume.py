"""Restore an opted-in team model through the harness resume protocol hook."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from octop_harness.protocols import register_protocol
from octop_harness.protocols.langgraph import LangGraphProtocol

from octop.infra.db.repos.threads import ThreadRow

TEAM_MODEL_RESUME_PROTOCOL = "octop-team-model-resume"
_TEAM_MODEL_KEY = "octop_team_model_override"


@dataclass(frozen=True)
class TeamModelResume:
    thread_id: str
    agent_id: str
    checkpoint_id: str
    interrupt_ids: tuple[str, ...]
    model: str
    user: str
    session_key: str | None
    peer_invoke_mode: str | None


_resume_snapshot: ContextVar[tuple[TeamModelResume, Callable[[str], ThreadRow | None]] | None] = (
    ContextVar("octop_team_model_resume", default=None)
)


async def read_team_model_resume(
    graph: Any,
    *,
    agent_id: str,
    thread_id: str,
    get_thread: Callable[[str], ThreadRow | None],
) -> TeamModelResume | None:
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    cfg = state.config.get("configurable") or {}
    meta = state.metadata or {}
    checkpoint_id = cfg.get("checkpoint_id")
    model = meta.get("model")
    override = meta.get(_TEAM_MODEL_KEY)
    user = meta.get("user")
    thread = get_thread(thread_id)
    if (
        thread is None
        or thread.agent_id != agent_id
        or str(thread.user_id) != str(user)
        or not state.interrupts
        or cfg.get("thread_id") != thread_id
        or cfg.get("checkpoint_ns", "") != ""
        or not isinstance(checkpoint_id, str)
        or not checkpoint_id
        or meta.get("agent_id") != agent_id
        or not isinstance(model, str)
        or not model.strip()
        or not isinstance(override, str)
        or override.strip() != model.strip()
        or isinstance(user, bool)
        or not isinstance(user, (str, int))
        or not str(user).isdigit()
        or int(user) <= 0
    ):
        return None
    session_key = meta.get("session_key")
    peer_mode = meta.get("peer_invoke_mode")
    return TeamModelResume(
        thread_id=thread_id,
        agent_id=agent_id,
        checkpoint_id=checkpoint_id,
        interrupt_ids=tuple(item.id for item in state.interrupts),
        model=model.strip(),
        user=str(user),
        session_key=session_key if isinstance(session_key, str) and session_key else None,
        peer_invoke_mode=peer_mode if peer_mode in ("async", "sync") else None,
    )


@contextmanager
def team_model_resume_scope(
    snapshot: TeamModelResume | None, get_thread: Callable[[str], ThreadRow | None]
) -> Iterator[None]:
    token = _resume_snapshot.set((snapshot, get_thread) if snapshot else None)
    try:
        yield
    finally:
        _resume_snapshot.reset(token)


class _TeamModelResumeProtocol(LangGraphProtocol):
    async def resume_stream(
        self,
        thread_id: str,
        decisions: list[dict[str, Any]],
        config: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        context = _resume_snapshot.get()
        if context is not None:
            snapshot, get_thread = context
            if snapshot.thread_id != thread_id:
                raise ValueError("Team model resume thread does not match the paused turn")
            current = await read_team_model_resume(
                self._graph,
                agent_id=snapshot.agent_id,
                thread_id=thread_id,
                get_thread=get_thread,
            )
            if current != snapshot:
                raise ValueError("Paused team turn changed before HITL resume")
            # The cached protocol is shared across threads; keep all turn data local.
            restored = {
                "model": snapshot.model,
                _TEAM_MODEL_KEY: snapshot.model,
                "agent_id": snapshot.agent_id,
                "user": snapshot.user,
            }
            if snapshot.session_key is not None:
                restored["session_key"] = snapshot.session_key
            if snapshot.peer_invoke_mode is not None:
                restored["peer_invoke_mode"] = snapshot.peer_invoke_mode
            config = {
                **config,
                "configurable": {**(config.get("configurable") or {}), **restored},
            }
        # Do not replay the saved checkpoint_id or restore permission metadata.
        async for chunk in super().resume_stream(thread_id, decisions, config, **kwargs):
            yield chunk


register_protocol(TEAM_MODEL_RESUME_PROTOCOL, _TeamModelResumeProtocol)
