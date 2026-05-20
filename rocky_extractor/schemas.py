from pydantic import BaseModel, Field
from typing import List, Optional

class RockyDialogue(BaseModel):
    chapter: Optional[str] = Field(None, description="The chapter name or number where the dialogue occurs.")
    scene_context: Optional[str] = Field(None, description="Brief description of the physical location, action, or context surrounding the dialogue.")
    speaker: Optional[str] = Field("Rocky", description="The speaker of this dialogue line, usually 'Rocky'.")
    dialogue: Optional[str] = Field(None, description="The exact text of Rocky's spoken dialogue (or its translation/musical representation).")
    emotion: Optional[str] = Field(None, description="Rocky's inferred or stated emotional state during this dialogue.")
    topic: Optional[str] = Field(None, description="The primary topic of conversation (e.g., fuel, Erid, math, physics, biology).")
    related_entities: List[str] = Field(default_factory=list, description="Any entities mentioned in the dialogue (e.g., xenonite, Astrophage, Adrian, Grace).")
    grace_response: Optional[str] = Field(None, description="Grace's immediate reaction or response to Rocky's dialogue.")

class DialogueExtractionResult(BaseModel):
    items: List[RockyDialogue] = Field(default_factory=list, description="List of Rocky dialogue records extracted from the chunk.")


class RockyMemory(BaseModel):
    event_id: Optional[str] = Field(None, description="A unique generated ID for the event, e.g., 'evt_c15_1'.")
    chapter: Optional[str] = Field(None, description="The chapter name or number where this memory/incident occurs.")
    event_summary: Optional[str] = Field(None, description="A detailed summary of the important event Rocky experiences or recounts.")
    participants: List[str] = Field(default_factory=list, description="Characters involved in this event (e.g., Rocky, Grace).")
    rocky_emotion: Optional[str] = Field(None, description="Rocky's emotional response or state during the event.")
    engineering_relevance: Optional[str] = Field(None, description="How the event relates to engineering, physics, science, or repairs (or null if not applicable).")
    knowledge_learned: List[str] = Field(default_factory=list, description="Facts or skills Rocky or Grace learned during this event.")
    important_objects: List[str] = Field(default_factory=list, description="Key objects involved (e.g., Blip-A, fuel tanks, Eridian tools, xenonite).")
    relationship_impact: Optional[str] = Field(None, description="The impact of this event on the relationship between Grace and Rocky.")
    confidence: Optional[str] = Field("high", description="Confidence level of this extraction, 'high', 'medium', or 'low'.")

class MemoryExtractionResult(BaseModel):
    items: List[RockyMemory] = Field(default_factory=list, description="List of memory/incident records extracted from the chunk.")


class RockyBehavior(BaseModel):
    chapter: Optional[str] = Field(None, description="The chapter name or number.")
    observation: Optional[str] = Field(None, description="Detailed physical or behavioral observation of Rocky (e.g., taps carapace, shakes carapace, moves hands, sleeps).")
    behavior_type: Optional[str] = Field(None, description="Category of behavior (e.g., gesture, physical movement, communication style, sleep, work pattern).")
    trigger: Optional[str] = Field(None, description="What triggered this behavioral response (e.g., danger, surprise, Grace's action, a scientific discovery).")
    interpretation: Optional[str] = Field(None, description="The meaning of this behavior (e.g., agreement, frustration, physical stress, thinking).")
    confidence: Optional[str] = Field("high", description="Confidence level of this extraction, 'high', 'medium', or 'low'.")

class BehaviorExtractionResult(BaseModel):
    items: List[RockyBehavior] = Field(default_factory=list, description="List of behavior records extracted from the chunk.")


class RockyKnowledge(BaseModel):
    chapter: Optional[str] = Field(None, description="The chapter name or number.")
    knowledge_type: Optional[str] = Field(None, description="Type of knowledge: 'fact', 'belief', 'misunderstanding', or 'learning'.")
    statement: Optional[str] = Field(None, description="The specific fact Rocky knows, belief he holds, thing he learned, or misunderstanding he has.")
    certainty: Optional[str] = Field(None, description="Certainty level: 'certain', 'uncertain', or 'erroneous'.")
    source_context: Optional[str] = Field(None, description="The textual context or quote supporting this statement.")

class KnowledgeExtractionResult(BaseModel):
    items: List[RockyKnowledge] = Field(default_factory=list, description="List of knowledge records extracted from the chunk.")


class RockyRelationship(BaseModel):
    chapter: Optional[str] = Field(None, description="The chapter name or number.")
    interaction_summary: Optional[str] = Field(None, description="Summary of the key interaction between Rocky and Grace.")
    relationship_state: Optional[str] = Field(None, description="The current state of their relationship (e.g., strangers, tentative allies, close friends, partners).")
    trust_level: Optional[str] = Field(None, description="Trust level description (e.g., none, growing, deep trust, absolute trust).")
    emotional_shift: Optional[str] = Field(None, description="Any emotional shift in either Rocky or Grace during this interaction.")
    important_quote: Optional[str] = Field(None, description="A key quote that illustrates the relationship dynamic in this scene.")

class RelationshipExtractionResult(BaseModel):
    items: List[RockyRelationship] = Field(default_factory=list, description="List of relationship records extracted from the chunk.")
