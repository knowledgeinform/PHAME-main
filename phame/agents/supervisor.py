# supervisor.py
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from librarian import librarian_agent, LibrarianDeps
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os


@dataclass
class SupervisorDeps:
    librarian_deps: LibrarianDeps
    librarian_history: list[ModelMessage] = field(default_factory=list)

agent_chat_model = OpenAIChatModel(
    "openai/gpt-oss-120b",   # model string passed through to the endpoint
    provider=OpenAIProvider(
        base_url=os.environ["PORTKEY_BASE_URL"],   # e.g. "https://api.portkey.ai/v1"
        api_key=os.environ["PORTKEY_API_KEY"],
    ),
)

supervisor_agent = Agent(
    agent_chat_model,
    deps_type=SupervisorDeps,
    system_prompt=(
        "You are the Supervisor.\n"
        "Delegate any knowledge-base / factual questions to the Librarian using `ask_librarian`.\n"
        "For planning, writing, coding, etc., you may answer directly or use other specialist tools.\n"
    ),
)

@supervisor_agent.tool
def ask_librarian(ctx: RunContext[SupervisorDeps], question: str) -> str:
    """Delegate a question to the librarian agent and return its answer."""
    print("Asking the librarian agent for facts on this prompt")
    result = librarian_agent.run_sync(
        question,
        deps=ctx.deps.librarian_deps,
        message_history=ctx.deps.librarian_history,
    )

    # Keep a separate conversation state for the librarian (optional but useful)
    ctx.deps.librarian_history[:] = result.all_messages()  # :contentReference[oaicite:1]{index=1}

    # PydanticAI uses `result.output` in the docs; fallback for older code:
    return getattr(result, "output", None) or getattr(result, "data", "")
