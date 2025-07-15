"""Models for multi-hop reasoning system"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ReasoningStepType(str, Enum):
    """Types of reasoning steps"""

    MEMORY_RETRIEVAL = "memory_retrieval"
    INFERENCE = "inference"
    ANALOGY = "analogy"
    PATTERN_MATCHING = "pattern_matching"
    TEMPORAL_REASONING = "temporal_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    SYNTHESIS = "synthesis"


class ConfidenceLevel(str, Enum):
    """Confidence levels for reasoning"""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReasoningStep(BaseModel):
    """A single step in the reasoning chain"""

    step_id: str = Field(..., description="Unique identifier for this step")
    step_type: ReasoningStepType = Field(..., description="Type of reasoning step")
    input_context: List[str] = Field(
        ..., description="Context used as input for this step"
    )
    reasoning_process: str = Field(
        ..., description="Description of the reasoning process"
    )
    output: str = Field(..., description="Output or conclusion from this step")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="Supporting evidence"
    )
    memory_references: List[str] = Field(
        default_factory=list, description="Referenced memory IDs"
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class ReasoningChain(BaseModel):
    """A complete reasoning chain from question to answer"""

    chain_id: str = Field(..., description="Unique identifier for this reasoning chain")
    question: str = Field(..., description="Original question")
    reasoning_steps: List[ReasoningStep] = Field(
        ..., description="Sequence of reasoning steps"
    )
    final_answer: str = Field(..., description="Final synthesized answer")
    overall_confidence: float = Field(
        ..., description="Overall confidence in the answer"
    )
    alternative_chains: List["ReasoningChain"] = Field(
        default_factory=list, description="Alternative reasoning paths"
    )
    total_memories_used: int = Field(
        ..., description="Total number of memories referenced"
    )
    reasoning_time_ms: int = Field(..., description="Time taken for reasoning")

    class Config:
        use_enum_values = True


class MemoryCluster(BaseModel):
    """A cluster of related memories"""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    theme: str = Field(..., description="Main theme or topic of the cluster")
    memories: List[Dict[str, Any]] = Field(..., description="Memories in this cluster")
    relationships: List[Dict[str, Any]] = Field(
        ..., description="Relationships between memories"
    )
    confidence: float = Field(..., description="Confidence in the clustering")
    temporal_span: Optional[Dict[str, datetime]] = Field(
        None, description="Time range of memories"
    )


class InferenceRule(BaseModel):
    """A rule for making inferences"""

    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable name")
    pattern: str = Field(..., description="Pattern this rule matches")
    inference_type: str = Field(..., description="Type of inference")
    confidence_factor: float = Field(
        ..., description="How much confidence this rule adds"
    )
    examples: List[str] = Field(
        default_factory=list, description="Example applications"
    )


class ContextWindow(BaseModel):
    """A window of context for reasoning"""

    window_id: str = Field(..., description="Unique window identifier")
    focal_memories: List[str] = Field(..., description="Primary memories in focus")
    supporting_memories: List[str] = Field(
        ..., description="Supporting context memories"
    )
    entities: List[str] = Field(..., description="Key entities in this context")
    relationships: List[Dict[str, Any]] = Field(..., description="Key relationships")
    temporal_context: Optional[Dict[str, Any]] = Field(
        None, description="Temporal information"
    )
    confidence: float = Field(..., description="Confidence in this context window")


class ReasoningResult(BaseModel):
    """Complete reasoning result with multiple possible answers"""

    question: str = Field(..., description="Original question")
    primary_chain: ReasoningChain = Field(
        ..., description="Most confident reasoning chain"
    )
    alternative_chains: List[ReasoningChain] = Field(
        default_factory=list, description="Alternative reasoning paths"
    )
    synthesis: str = Field(..., description="Synthesized answer considering all chains")
    confidence_distribution: Dict[str, float] = Field(
        ..., description="Confidence across different aspects"
    )
    contradictions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified contradictions"
    )
    gaps: List[str] = Field(
        default_factory=list, description="Information gaps identified"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )
    reasoning_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the reasoning process"
    )
