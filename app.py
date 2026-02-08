from context_builder import ContextBuilder
from memory import MemoryStore
from retriever import VectorRetriever


def main():
    """
    CLI entry point for the Context-Window-Aware RAG demo.

    This function orchestrates:
    - user input (query)
    - vector retrieval
    - short-term memory collection
    - context assembly
    - overflow reporting
    """

    print("\n=== CONTEXT-WINDOW-AWARE RAG DEMO ===\n")

    # ------------------------------------------------------------------
    # User goal / query
    # ------------------------------------------------------------------
    query = "How does Salesforce store and manage leads?"

    # ------------------------------------------------------------------
    # Vector retrieval
    # Retrieves more content than can fit to demonstrate overflow handling
    # ------------------------------------------------------------------
    retriever = VectorRetriever("corpus")
    retrieved_chunks = retriever.retrieve(query, top_k=8)

    # ------------------------------------------------------------------
    # Memory store
    # Intentionally add multiple items to approach or exceed memory budget
    # ------------------------------------------------------------------
    memory = MemoryStore(token_budget=55)
    memory.add("User previously asked about Salesforce lead syncing.")
    memory.add("User prefers short summaries.")
    memory.add("User is preparing for an interview.")
    memory.add("User wants step-by-step explanations.")

    # ------------------------------------------------------------------
    # Context builder
    # Responsible for enforcing budgets and assembling final prompt
    # ------------------------------------------------------------------
    builder = ContextBuilder()

    # System-level instructions (highest priority)
    instructions = builder.build_instructions(
        "You are a helpful assistant. Answer clearly, concisely, and accurately."
    )

    # User goal / intent
    goal = builder.build_goal(query)

    # Short-term conversational memory
    memory_section = builder.build_memory(memory.items)

    # Retrieved external knowledge
    retrieval_section = builder.build_retrieval(retrieved_chunks)

    # ------------------------------------------------------------------
    # Assemble final context window in priority order
    # ------------------------------------------------------------------
    result = builder.assemble([
        instructions,
        goal,
        memory_section,
        retrieval_section
    ])

    # ------------------------------------------------------------------
    # Explicit overflow reporting
    # ------------------------------------------------------------------
    print("\n--- OVERFLOW REPORT ---")

    if memory_section.overflowed:
        print("MEMORY overflowed: oldest items dropped")

    if retrieval_section.overflowed:
        print(
            f"RETRIEVAL overflowed: dropped "
            f"{retrieval_section.dropped_items} chunks"
        )

    # ------------------------------------------------------------------
    # Display final assembled context
    # ------------------------------------------------------------------
    print("\n--- FINAL CONTEXT ---")
    print(result["context"])

    # ------------------------------------------------------------------
    # Display token usage per section for transparency
    # ------------------------------------------------------------------
    print("\n--- TOKEN REPORT ---")
    for section, info in result["report"].items():
        print(section, info)


if __name__ == "__main__":
    main()
