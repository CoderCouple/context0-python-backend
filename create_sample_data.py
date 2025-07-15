#!/usr/bin/env python3
"""Create comprehensive sample data for testing multi-hop reasoning"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import List

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api.v1.request.memory_request import MemoryRecordInput
from app.common.enum.memory import MemoryType
from app.memory.engine.memory_engine import MemoryEngine


async def create_comprehensive_sample_data():
    """Create comprehensive sample data with interconnected memories for testing"""
    print("🌱 Creating comprehensive sample data for multi-hop reasoning...")

    # Initialize memory engine
    engine = MemoryEngine.get_instance()
    await engine.initialize()

    # User profile - use fresh user ID to avoid conflicts with old data
    user_id = "john-doe-fresh"
    session_id = "comprehensive-test-session"

    # Sample data with interconnected memories for advanced reasoning
    sample_memories = [
        # Educational Background & Skills
        {
            "text": "I graduated from MIT in 2012 with a Computer Science degree, specializing in AI and machine learning",
            "memory_type": MemoryType.SEMANTIC_MEMORY,
            "tags": [
                "education",
                "MIT",
                "computer-science",
                "AI",
                "machine-learning",
                "2012",
            ],
            "metadata": {
                "institution": "MIT",
                "degree": "CS",
                "graduation_year": 2012,
                "specialization": "AI",
            },
        },
        {
            "text": "During college at MIT, I worked with Professor Regina Barzilay on natural language processing research, focusing on sentiment analysis",
            "memory_type": MemoryType.EPISODIC_MEMORY,
            "tags": [
                "MIT",
                "research",
                "NLP",
                "sentiment-analysis",
                "professor",
                "Regina-Barzilay",
            ],
            "metadata": {
                "research_area": "NLP",
                "professor": "Regina Barzilay",
                "focus": "sentiment analysis",
            },
        },
        {
            "text": "My senior thesis at MIT was titled 'Hierarchical Attention Networks for Document Classification' and received summa cum laude",
            "memory_type": MemoryType.SEMANTIC_MEMORY,
            "tags": [
                "MIT",
                "thesis",
                "attention-networks",
                "document-classification",
                "summa-cum-laude",
            ],
            "metadata": {
                "thesis_title": "Hierarchical Attention Networks for Document Classification",
                "honor": "summa cum laude",
            },
        },
        # Professional Experience
        {
            "text": "After MIT, I joined Google in 2012 as a Software Engineer, working on search ranking algorithms in the Information Retrieval team",
            "memory_type": MemoryType.EPISODIC_MEMORY,
            "tags": [
                "Google",
                "software-engineer",
                "search",
                "ranking",
                "information-retrieval",
                "2012",
            ],
            "metadata": {
                "company": "Google",
                "role": "Software Engineer",
                "team": "Information Retrieval",
                "start_year": 2012,
            },
        },
        {
            "text": "During my time at Google, I contributed to the development of RankBrain, Google's machine learning system for search",
            "memory_type": MemoryType.PROCEDURAL_MEMORY,
            "tags": [
                "Google",
                "RankBrain",
                "machine-learning",
                "search",
                "contribution",
            ],
            "metadata": {
                "project": "RankBrain",
                "contribution_type": "development",
                "impact": "search improvement",
            },
        },
        {
            "text": "In 2015, I was promoted to Senior Software Engineer at Google and led a team of 5 engineers working on query understanding",
            "memory_type": MemoryType.EPISODIC_MEMORY,
            "tags": [
                "Google",
                "promotion",
                "senior-engineer",
                "leadership",
                "query-understanding",
                "2015",
            ],
            "metadata": {
                "promotion_year": 2015,
                "team_size": 5,
                "focus_area": "query understanding",
            },
        },
        {
            "text": "I left Google in 2018 to join OpenAI as a Research Scientist, focusing on language model development and safety",
            "memory_type": MemoryType.EPISODIC_MEMORY,
            "tags": [
                "OpenAI",
                "research-scientist",
                "language-models",
                "safety",
                "2018",
                "career-change",
            ],
            "metadata": {
                "company": "OpenAI",
                "role": "Research Scientist",
                "focus": "language models",
                "start_year": 2018,
            },
        },
        # Technical Skills & Knowledge
        {
            "text": "I'm expert-level in Python, TensorFlow, PyTorch, and have published 12 papers on neural network architectures",
            "memory_type": MemoryType.PROCEDURAL_MEMORY,
            "tags": [
                "Python",
                "TensorFlow",
                "PyTorch",
                "neural-networks",
                "publications",
                "expert",
            ],
            "metadata": {
                "programming_languages": ["Python"],
                "frameworks": ["TensorFlow", "PyTorch"],
                "publications": 12,
            },
        },
        {
            "text": "My most cited paper 'Attention Is All You Need for Document Retrieval' has over 500 citations and introduced novel attention mechanisms",
            "memory_type": MemoryType.SEMANTIC_MEMORY,
            "tags": [
                "research",
                "attention",
                "document-retrieval",
                "citations",
                "innovation",
            ],
            "metadata": {
                "paper_title": "Attention Is All You Need for Document Retrieval",
                "citations": 500,
                "innovation": "attention mechanisms",
            },
        },
        # Personal Projects & Interests
        {
            "text": "I built an open-source library called 'MemoryBank' for efficient vector search, which has 10k GitHub stars",
            "memory_type": MemoryType.PROCEDURAL_MEMORY,
            "tags": [
                "open-source",
                "MemoryBank",
                "vector-search",
                "GitHub",
                "programming",
            ],
            "metadata": {
                "project": "MemoryBank",
                "type": "library",
                "stars": 10000,
                "purpose": "vector search",
            },
        },
        {
            "text": "I'm passionate about photography and have won 3 awards in landscape photography competitions",
            "memory_type": MemoryType.EPISODIC_MEMORY,
            "tags": ["photography", "landscape", "awards", "hobby", "competition"],
            "metadata": {"hobby": "photography", "specialty": "landscape", "awards": 3},
        },
        {
            "text": "I play classical guitar and have performed at 5 local venues, combining my love for music with technical precision",
            "memory_type": MemoryType.EPISODIC_MEMORY,
            "tags": ["guitar", "classical", "music", "performance", "venues"],
            "metadata": {
                "instrument": "classical guitar",
                "performances": 5,
                "skill": "technical precision",
            },
        },
        # Recent Work & Future Goals
        {
            "text": "Currently at OpenAI, I'm working on developing more interpretable AI systems and published a recent paper on mechanistic interpretability",
            "memory_type": MemoryType.PROCEDURAL_MEMORY,
            "tags": [
                "OpenAI",
                "interpretability",
                "AI-systems",
                "current-work",
                "research",
            ],
            "metadata": {
                "current_focus": "interpretable AI",
                "recent_work": "mechanistic interpretability",
                "status": "current",
            },
        },
        {
            "text": "My long-term goal is to start an AI safety research institute focused on alignment problems in large language models",
            "memory_type": MemoryType.WORKING_MEMORY,
            "tags": [
                "future-goals",
                "AI-safety",
                "research-institute",
                "alignment",
                "LLMs",
            ],
            "metadata": {
                "goal_type": "long-term",
                "focus": "AI safety",
                "target": "research institute",
            },
        },
        # Meta-cognition & Learning
        {
            "text": "I've learned that my systematic approach to problem-solving, developed during my MIT education, has been key to my research success",
            "memory_type": MemoryType.META_MEMORY,
            "tags": [
                "meta-learning",
                "problem-solving",
                "MIT",
                "systematic-approach",
                "research-success",
            ],
            "metadata": {
                "learning_type": "meta-cognitive",
                "insight": "systematic approach",
                "origin": "MIT education",
            },
        },
        {
            "text": "I notice that I work best in collaborative environments where I can combine technical depth with creative exploration",
            "memory_type": MemoryType.META_MEMORY,
            "tags": [
                "self-awareness",
                "collaboration",
                "technical-depth",
                "creativity",
                "work-style",
            ],
            "metadata": {
                "work_preference": "collaborative",
                "strength": "technical + creative",
                "awareness": "optimal conditions",
            },
        },
        # Emotional & Social Context
        {
            "text": "I felt most fulfilled when my RankBrain work at Google directly improved search results for millions of users",
            "memory_type": MemoryType.EMOTIONAL_MEMORY,
            "tags": [
                "fulfillment",
                "RankBrain",
                "Google",
                "impact",
                "users",
                "satisfaction",
            ],
            "metadata": {
                "emotion": "fulfillment",
                "trigger": "user impact",
                "project": "RankBrain",
            },
        },
        {
            "text": "The transition from Google to OpenAI was emotionally challenging but ultimately rewarding as it aligned with my values around AI safety",
            "memory_type": MemoryType.EMOTIONAL_MEMORY,
            "tags": [
                "career-transition",
                "challenging",
                "rewarding",
                "values",
                "AI-safety",
            ],
            "metadata": {
                "transition": "Google to OpenAI",
                "emotion": "challenging then rewarding",
                "motivation": "values alignment",
            },
        },
        # Declarative Knowledge
        {
            "text": "The Transformer architecture, introduced in 'Attention Is All You Need', revolutionized NLP by replacing recurrent layers with self-attention",
            "memory_type": MemoryType.DECLARATIVE_MEMORY,
            "tags": [
                "Transformer",
                "attention",
                "NLP",
                "self-attention",
                "architecture",
                "knowledge",
            ],
            "metadata": {
                "concept": "Transformer architecture",
                "innovation": "self-attention",
                "impact": "NLP revolution",
            },
        },
        {
            "text": "Vector databases like Pinecone use approximate nearest neighbor search algorithms like HNSW to enable fast similarity search at scale",
            "memory_type": MemoryType.DECLARATIVE_MEMORY,
            "tags": [
                "vector-databases",
                "Pinecone",
                "ANN",
                "HNSW",
                "similarity-search",
                "scale",
            ],
            "metadata": {
                "technology": "vector databases",
                "algorithm": "HNSW",
                "purpose": "fast similarity search",
            },
        },
    ]

    print(f"Creating {len(sample_memories)} interconnected memories...")

    # Create memories with proper timestamps for temporal reasoning
    base_time = datetime.now() - timedelta(days=365)  # Start a year ago

    for i, memory_data in enumerate(sample_memories):
        # Add temporal distribution
        created_time = base_time + timedelta(days=i * 15)  # Spread over time

        record_input = MemoryRecordInput(
            user_id=user_id,
            session_id=session_id,
            text=memory_data["text"],
            memory_type=memory_data["memory_type"],
            tags=memory_data["tags"],
            metadata=memory_data["metadata"],
            scope="private",
        )

        try:
            response = await engine.add_memory(record_input)
            if response.success:
                print(
                    f"✅ Created memory {i+1}: {memory_data['memory_type'].value} - {memory_data['text'][:60]}..."
                )
            else:
                print(f"❌ Failed to create memory {i+1}: {response.message}")
        except Exception as e:
            print(f"❌ Error creating memory {i+1}: {e}")

    print("\n🎯 Sample data creation completed!")
    print("📊 Data includes:")
    print("   - Educational background (MIT, CS degree)")
    print("   - Professional experience (Google, OpenAI)")
    print("   - Technical skills and research contributions")
    print("   - Personal interests and hobbies")
    print("   - Meta-cognitive insights")
    print("   - Emotional contexts")
    print("   - Declarative knowledge")
    print("\n🔗 These memories are interconnected for complex multi-hop reasoning:")
    print("   - MIT education → Google career → OpenAI transition")
    print("   - Research skills → Publications → Open source projects")
    print("   - Technical expertise → Practical applications → Future goals")
    print("   - Personal insights → Work preferences → Career choices")


if __name__ == "__main__":
    asyncio.run(create_comprehensive_sample_data())
