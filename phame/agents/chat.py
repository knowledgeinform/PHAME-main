# chat.py
from pydantic_ai.messages import ModelMessage
from supervisor import supervisor_agent, SupervisorDeps
from librarian import LibrarianDeps

# from phame.haystack.agent_calls_rag import build_rag_pipeline
from phame.haystack.trusted_references_rag import make_chroma_document_store, build_rag_pipeline

# textbook_rag = build_rag_pipeline()
CHROMA_PERSIST = "./chroma_db"
EMBED_MODEL = "intfloat/e5-large-v2"
document_store = make_chroma_document_store(persist_path=CHROMA_PERSIST)
textbook_rag = build_rag_pipeline(document_store, EMBED_MODEL)

deps = SupervisorDeps(librarian_deps=LibrarianDeps(textbook_rag=textbook_rag))
supervisor_history: list[ModelMessage] = []

while True:
    user = input("you> ").strip()
    if user.lower() in {"exit", "quit"}:
        break

    result = supervisor_agent.run_sync(user, deps=deps, message_history=supervisor_history)
    print("supervisor>", getattr(result, "output", None) or getattr(result, "data", ""))

    # Persist supervisor conversation
    supervisor_history = result.all_messages()  # :contentReference[oaicite:3]{index=3}
