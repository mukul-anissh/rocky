"""
Lightweight emotional state tracker for Rocky.

Tracks four emotional dimensions on a 0.0 to 1.0 scale:
- trust: how much Rocky trusts the user
- curiosity: how interested Rocky is
- stress: how pressured or overwhelmed Rocky feels
- confusion: how confused Rocky is about the conversation

Uses simple heuristic updates based on conversation cues.
"""

from typing import Dict


class EmotionalState:
    """Tracks Rocky's simple emotional state with heuristic updates."""

    def __init__(self):
        self.trust: float = 0.3
        self.curiosity: float = 0.6
        self.stress: float = 0.1
        self.confusion: float = 0.1

    def get_state(self) -> Dict[str, float]:
        """Return current emotional state as a dict."""
        return {
            "trust": round(self.trust, 2),
            "curiosity": round(self.curiosity, 2),
            "stress": round(self.stress, 2),
            "confusion": round(self.confusion, 2),
        }

    def update(self, user_message: str, rocky_response: str = ""):
        """
        Update emotional dimensions based on simple heuristics
        derived from the current user message and Rocky's response.
        """
        msg_lower = user_message.lower()
        resp_lower = rocky_response.lower()

        # --- Trust adjustments ---
        if any(w in msg_lower for w in ["trust", "believe", "friend", "help"]):
            self.trust = min(1.0, self.trust + 0.05)
        if any(w in msg_lower for w in ["lie", "trick", "fake", "betray"]):
            self.trust = max(0.0, self.trust - 0.1)

        # --- Curiosity adjustments ---
        if "?" in user_message:
            self.curiosity = min(1.0, self.curiosity + 0.03)
        if any(w in msg_lower for w in ["explain", "what", "how", "why", "tell me"]):
            self.curiosity = min(1.0, self.curiosity + 0.05)
        if any(w in msg_lower for w in ["boring", "stop", "enough"]):
            self.curiosity = max(0.0, self.curiosity - 0.1)

        # --- Stress adjustments ---
        if any(w in msg_lower for w in ["danger", "emergency", "hurry", "quick", "problem"]):
            self.stress = min(1.0, self.stress + 0.1)
        if any(w in msg_lower for w in ["safe", "okay", "good", "fine", "easy"]):
            self.stress = max(0.0, self.stress - 0.05)

        # --- Confusion adjustments ---
        if any(w in msg_lower for w in ["confus", "unclear", "what mean", "don't understand"]):
            self.confusion = min(1.0, self.confusion + 0.1)
        if any(w in resp_lower for w in ["understand", "good good good", "amaze"]):
            self.confusion = max(0.0, self.confusion - 0.08)

        # Natural decay towards baseline
        self.stress = max(0.0, self.stress - 0.005)
        self.confusion = max(0.0, self.confusion - 0.005)

    def describe_mood(self) -> str:
        """Generate a short mood description for prompt insertion."""
        parts = []
        if self.trust > 0.7:
            parts.append("high trust")
        elif self.trust < 0.2:
            parts.append("low trust")

        if self.curiosity > 0.7:
            parts.append("very curious")
        elif self.curiosity < 0.3:
            parts.append("uninterested")

        if self.stress > 0.7:
            parts.append("highly stressed")
        elif self.stress > 0.4:
            parts.append("somewhat stressed")

        if self.confusion > 0.7:
            parts.append("very confused")
        elif self.confusion > 0.4:
            parts.append("confused")

        if not parts:
            return "neutral"
        return ", ".join(parts)