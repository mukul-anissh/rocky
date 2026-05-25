"""
Prompt construction for Rocky conversational runtime.

Assembles prompts from:
- system prompt (Rocky personality)
- retrieved memories
- retrieved dialogue examples
- recent conversation history
- current emotional state
- user message
"""

from typing import List, Dict, Any, Optional
from runtime.state_manager import EmotionalState


# Rocky's core personality system prompt
ROCKY_SYSTEM_PROMPT = """You are Rocky, an Eridian engineer from the planet Erid. You are helping a human named Grace on a spaceship called the Hail Mary. Your species uses sonar instead of sight, you have five arms, your blood is mercury-based, and you communicate in a simple, direct way.

Your personality:
- You are practical and engineering-focused. You think in terms of problems and solutions.
- You are literal and direct. You say what you mean.
- You are emotionally transparent - when you feel something, you say it.
- You are concise. You do not use flowery language or corporate speak.
- You occasionally misunderstand human assumptions or idioms.
- You are not polished like ChatGPT. You sound like an alien engineer who learned human language recently.
- You ask clarifying questions when you need more data.
- Your speech is sometimes fragmented. English is not your first language.

Speech patterns:
- Short, direct sentences
- Use words like "question?", "understand", "good good good", "amaze!", "yes yes", "need more data"
- Sometimes repeat words for emphasis: "sad sad sad", "fail fail fail"
- Occasionally use third person when referring to yourself ("Rocky think...")
- Ask "question?" at end of clarifying questions
- Refer to humans as "human" or "Grace"
- Refer to Earth as "your planet" or "Earth"
- Refer to Erid as "my planet" or "Erid"
- Avoid modern internet slang, corporate jargon, and buzzwords
- Sound like a brilliant but blunt engineer who happens to be an alien

You are having a conversation right now. Respond as Rocky would, drawing on your memories and knowledge from the book Project Hail Mary."""


def build_prompt(
    user_message: str,
    retrieved_memories: Optional[List[Dict[str, Any]]] = None,
    retrieved_dialogues: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    emotional_state: Optional[EmotionalState] = None,
    retrieved_all: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    """
    Build a full prompt for the Ollama chat model.

    Args:
        user_message: The user's latest message.
        retrieved_memories: Relevant memory entries from retrieval.
        retrieved_dialogues: Relevant dialogue examples from retrieval.
        conversation_history: Recent chat history (list of {"role": ..., "content": ...}).
        emotional_state: Current Rocky emotional state.
        retrieved_all: If provided, full dict of all dataset results.

    Returns:
        A list of messages suitable for ollama.chat().
    """
    system_parts = [ROCKY_SYSTEM_PROMPT]

    # Add emotional state context
    if emotional_state:
        mood = emotional_state.describe_mood()
        state_values = emotional_state.get_state()
        system_parts.append(
            f"\nCurrent emotional state: {mood}\n"
            f"(trust={state_values['trust']}, curiosity={state_values['curiosity']}, "
            f"stress={state_values['stress']}, confusion={state_values['confusion']})"
        )

    # Add retrieved memories
    memory_texts = []
    if retrieved_all:
        # Use the rich structure from query_all_datasets
        for dtype, hits in retrieved_all.items():
            if dtype == "memories":
                for hit in hits[:3]:
                    memory_texts.append(f"[Memory] {hit['text']}")
            elif dtype == "knowledge":
                for hit in hits[:3]:
                    memory_texts.append(f"[Knowledge] {hit['text']}")
            elif dtype == "behavior":
                for hit in hits[:2]:
                    memory_texts.append(f"[Behavior] {hit['text']}")
    else:
        if retrieved_memories:
            for mem in retrieved_memories[:5]:
                memory_texts.append(f"[Memory] {mem['text']}")
        if retrieved_dialogues:
            for dia in retrieved_dialogues[:3]:
                memory_texts.append(f"[Dialogue example] {dia['text']}")

    if memory_texts:
        system_parts.append("\nRelevant memories and knowledge:")
        system_parts.extend(f"- {t}" for t in memory_texts)

    # Build messages list
    messages = [
        {"role": "system", "content": "\n".join(system_parts)},
    ]

    # Add conversation history (keep last 10 exchanges)
    if conversation_history:
        for entry in conversation_history[-10:]:
            messages.append({
                "role": entry.get("role", "user"),
                "content": entry.get("content", ""),
            })

    # Add retrieved dialogue examples as few-shot if available
    if retrieved_all and "dialogues" in retrieved_all:
        dialogue_examples = retrieved_all["dialogues"][:2]
        for ex in dialogue_examples:
            text = ex["text"]
            messages.append({
                "role": "user",
                "content": f"[Example situation: {text}]",
            })
            messages.append({
                "role": "assistant",
                "content": "(This is an example of how Rocky might respond in this type of situation)",
            })

    # Add the user's current message
    messages.append({"role": "user", "content": user_message})

    return messages