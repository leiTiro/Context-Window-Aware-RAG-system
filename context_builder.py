from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class ContextSection:
    """
    Represents a single section of the final LLM context.

    Each section has:
    - a fixed token budget
    - a priority (lower number = higher priority)
    - overflow metadata for transparency and debugging
    """
    name: str                 # Section name (e.g. INSTRUCTIONS, MEMORY)
    content: str              # Final text included in this section
    token_budget: int         # Maximum allowed tokens for the section
    priority: int             # Assembly order (lower = earlier in prompt)
    overflowed: bool          # Whether content exceeded the budget
    tokens_used: int          # Estimated tokens used
    dropped_items: int = 0    # Number of items dropped due to overflow


class ContextBuilder:
    """
    Responsible for assembling the final LLM context window.

    This class enforces:
    - strict token budgets per section
    - section-specific overflow strategies
    - deterministic ordering of context sections
    """

    def __init__(self):
        """
        Define hard token budgets for each context section.

        Budgets are intentionally explicit to make trade-offs visible.
        """
        self.budgets = {
            "instructions": 255,  # System rules
            "goal": 1500,         # User intent / task
            "memory": 55,         # Short-term conversational memory
            "retrieval": 550,     # Retrieved external knowledge

            # Uncomment to force retrieval overflow for demo purposes
            # "retrieval": 200,
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token usage using a simple heuristic.

        This does not need to be perfectly accurate;
        consistency is more important than precision.
        """
        return int(len(text.split()) * 1.3)

    def build_instructions(self, text: str) -> ContextSection:
        """
        Build the INSTRUCTIONS section.

        Strategy:
        - Highest priority section
        - Truncated hard if budget is exceeded
        """
        tokens = self.estimate_tokens(text)
        overflowed = tokens > self.budgets["instructions"]

        if overflowed:
            words = text.split()
            max_words = int(self.budgets["instructions"] / 1.3)
            text = " ".join(words[:max_words])
            tokens = self.estimate_tokens(text)

        return ContextSection(
            name="INSTRUCTIONS",
            content=text,
            token_budget=self.budgets["instructions"],
            priority=1,
            overflowed=overflowed,
            tokens_used=tokens
        )

    def build_goal(self, text: str) -> ContextSection:
        """
        Build the GOAL section.

        Strategy:
        - Preserve user intent as much as possible
        - Only flagged as overflowed if extremely long
        """
        tokens = self.estimate_tokens(text)
        overflowed = tokens > self.budgets["goal"]

        return ContextSection(
            name="GOAL",
            content=text,
            token_budget=self.budgets["goal"],
            priority=2,
            overflowed=overflowed,
            tokens_used=tokens
        )

    def build_memory(self, memory_items: List[str]) -> ContextSection:
        """
        Build the MEMORY section.

        Strategy:
        - Memory is temporal
        - Newest items are prioritised
        - Oldest items are dropped first when budget is exceeded
        """
        used = 0
        selected = []

        # Iterate from newest → oldest
        for item in reversed(memory_items):
            tokens = self.estimate_tokens(item)
            if used + tokens > self.budgets["memory"]:
                break
            selected.insert(0, item)
            used += tokens

        overflowed = len(selected) < len(memory_items)

        return ContextSection(
            name="MEMORY",
            content="\n".join(selected),
            token_budget=self.budgets["memory"],
            priority=3,
            overflowed=overflowed,
            tokens_used=used,
            dropped_items=len(memory_items) - len(selected)
        )

    def build_retrieval(
        self,
        retrieved_chunks: List[Tuple[str, float]]
    ) -> ContextSection:
        """
        Build the RETRIEVAL section.

        Strategy:
        - Chunks are assumed to be pre-ranked by relevance
        - Highest relevance chunks are included first
        - Lowest relevance chunks are dropped when budget is exceeded
        """
        used = 0
        included = []
        dropped = []

        for text, score in retrieved_chunks:
            tokens = self.estimate_tokens(text)
            if used + tokens <= self.budgets["retrieval"]:
                included.append(text)
                used += tokens
            else:
                dropped.append(text)

        overflowed = len(dropped) > 0

        return ContextSection(
            name="RETRIEVAL",
            content="\n".join(included),
            token_budget=self.budgets["retrieval"],
            priority=4,
            overflowed=overflowed,
            tokens_used=used,
            dropped_items=len(dropped)
        )

    def assemble(self, sections: List[ContextSection]) -> Dict:
        """
        Assemble the final context window.

        Sections are sorted by priority and concatenated.
        A structured report is returned for transparency.
        """
        sections = sorted(sections, key=lambda s: s.priority)

        final_context = ""
        report = {}

        for section in sections:
            final_context += f"\n### {section.name}\n{section.content}\n"

            report[section.name] = {
                "tokens_used": section.tokens_used,
                "budget": section.token_budget,
                "overflowed": section.overflowed,
                "dropped_items": section.dropped_items
            }

        return {
            "context": final_context.strip(),
            "report": report
        }
