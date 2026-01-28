# librarian.py
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from haystack import Pipeline
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os

@dataclass
class LibrarianDeps:
    textbook_rag: Pipeline  # you can add more pipelines later

agent_chat_model = OpenAIChatModel(
    "openai/gpt-oss-120b",   # model string passed through to the endpoint
    provider=OpenAIProvider(
        base_url=os.environ["PORTKEY_BASE_URL"],   # e.g. "https://api.portkey.ai/v1"
        api_key=os.environ["PORTKEY_API_KEY"],
    ),
)

librarian_agent = Agent(
    agent_chat_model,
    deps_type=LibrarianDeps,
    system_prompt=(
        "You are the Librarian.\n"
        "Use your tools (RAG pipelines) to answer knowledge-base questions.\n"
        "If the KB doesn't contain the answer, say you don't know."
    ),
)

@librarian_agent.tool
def kb_basic(ctx: RunContext[LibrarianDeps], question: str) -> str:
    """Answer using the basic Haystack RAG pipeline."""
    p = ctx.deps.textbook_rag
    out = p.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},  # matches your template variable
        "answer_builder": {"query": question},     # AnswerBuilder expects query
    })
    return out["first_answer"]["answer"]
