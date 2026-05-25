"""
Rocky conversational chat runtime.

Terminal loop:
    User input -> retrieval -> prompt assembly -> Ollama generation -> print response

Usage:
    python -m runtime.rocky_chat
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class RockyChat:
    """Main chat runtime for Rocky conversational AI."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        temperature: float = 0.5,
        top_p: float = 0.9,
        repeat_penalty: float = 1.15,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize components
        from runtime.state_manager import EmotionalState
        from runtime.retrieve import Retriever

        self.emotional_state = EmotionalState()
        self.retriever = Retriever()
        self._check_collection()

        logger.info(f"Rocky chat initialized (model={model})")

    def _check_collection(self):
        """Warn if no ChromaDB collection was loaded."""
        if self.retriever.collection is None:
            print(
                "WARNING: ChromaDB collection not found. Run 'python -m runtime.embed' first.\n"
            )

    def _generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Send messages to Ollama and get response."""
        try:
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "repeat_penalty": self.repeat_penalty,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

    def respond(self, user_message: str) -> Optional[str]:
        """
        Process a user message through the full pipeline:
        retrieval -> prompt assembly -> generation -> state update
        """
        # 1. Retrieve relevant context
        retrieved_all = self.retriever.query_all_datasets(
            user_message, results_per_dataset=3
        )

        # Extract specific types for the prompt builder
        retrieved_memories = retrieved_all.get("memories", [])
        retrieved_dialogues = retrieved_all.get("dialogues", [])

        # 2. Build prompt
        from runtime.prompt_builder import build_prompt

        messages = build_prompt(
            user_message=user_message,
            retrieved_memories=retrieved_memories,
            retrieved_dialogues=retrieved_dialogues,
            conversation_history=self.conversation_history,
            emotional_state=self.emotional_state,
            retrieved_all=retrieved_all,
        )

        # 3. Generate response
        rocky_response = self._generate(messages)

        if rocky_response is None:
            return None

        # 4. Update emotional state based on the exchange
        self.emotional_state.update(user_message, rocky_response)

        # 5. Store in conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": rocky_response})

        return rocky_response


def print_welcome():
    """Print welcome banner."""
    border = "=" * 60
    print(f"\n{border}")
    print("  Rocky - Eridian Engineer Chat Interface")
    print(f"{border}")
    print("  Talk to Rocky from Project Hail Mary.")
    print("  Type 'quit' or 'exit' to end the conversation.")
    print(f"  Type 'reset' to clear conversation history.")
    print(f"{border}\n")


def print_state_indicators(chat: RockyChat):
    """Print current emotional state indicators."""
    state = chat.emotional_state.get_state()
    indicators = []
    for key, val in state.items():
        bar_len = 10
        filled = int(val * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        indicators.append(f"{key:>10}: [{bar}] {val:.2f}")
    print("  " + " | ".join(indicators))


def main():
    """Run the terminal chat loop."""
    print_welcome()

    chat = RockyChat()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Rocky: Goodbye, human. Fist my bump!")
            break

        if user_input.lower() == "reset":
            chat.conversation_history.clear()
            chat.emotional_state = chat.emotional_state.__class__()
            print("Rocky: Conversation reset. Start fresh. Good good good.")
            continue

        if user_input.lower() == "/status":
            print_state_indicators(chat)
            continue

        # Generate response
        print("Rocky: ", end="", flush=True)
        response = chat.respond(user_input)

        if response:
            print(response)
        else:
            print("[Error generating response. Check that Ollama is running.]")


if __name__ == "__main__":
    main()