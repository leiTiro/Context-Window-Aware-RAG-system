class MemoryStore:
    """
    Simple short-term memory store for the agent.

    This memory represents user-specific, conversational context
    (not factual knowledge). It is intentionally small and volatile
    to reflect how memory should behave in an agent system.
    """

    def __init__(self, token_budget: int):
        """
        Initialize the memory store.

        Args:
            token_budget (int): Maximum number of tokens allowed
                                for the memory section.
        """
        self.token_budget = token_budget  # Hard token limit for memory
        self.items = []                   # Stored memory entries (oldest → newest)

    def add(self, text: str):
        """
        Add a new memory entry.

        New memory items are appended to the end of the list.
        When memory is later assembled, newer entries are prioritised
        and older ones may be dropped if the token budget is exceeded.

        Args:
            text (str): A short, user-specific memory string.
        """
        self.items.append(text)
