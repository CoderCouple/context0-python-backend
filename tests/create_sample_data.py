#!/usr/bin/env python3
"""
Sample Data Creator for Multi-Hop Reasoning Testing
Creates a rich, interconnected dataset perfect for testing multi-hop reasoning capabilities
"""

import asyncio
import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import aiohttp

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_USER_ID = "john-doe"
SAMPLE_SESSION_ID = "sample-session"


class SampleDataCreator:
    """Creates comprehensive sample data for multi-hop reasoning tests"""

    def __init__(self):
        self.session = None
        self.created_memories = []
        self.memory_categories = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    def get_comprehensive_sample_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate comprehensive, interconnected sample memories"""

        return {
            "personal_background": [
                {
                    "text": "I was born on July 28, 1990, in Seattle, Washington",
                    "memory_type": "semantic_memory",
                    "tags": ["personal", "birth", "seattle", "family"],
                    "scope": "identity",
                },
                {
                    "text": "My parents are Michael (engineer) and Susan (teacher), they divorced when I was 16",
                    "memory_type": "semantic_memory",
                    "tags": ["family", "parents", "divorce", "engineer", "teacher"],
                    "scope": "family",
                },
                {
                    "text": "I have a younger sister Emma who is now a doctor in Portland",
                    "memory_type": "semantic_memory",
                    "tags": ["family", "sister", "doctor", "portland"],
                    "scope": "family",
                },
                {
                    "text": "Growing up, I was always fascinated by how things work - I took apart every electronic device I could find",
                    "memory_type": "episodic_memory",
                    "tags": ["childhood", "curiosity", "electronics", "personality"],
                    "scope": "identity",
                },
            ],
            "education_journey": [
                {
                    "text": "I studied Computer Science at University of Washington from 2008-2012",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "education",
                        "computer_science",
                        "university",
                        "washington",
                    ],
                    "scope": "education",
                },
                {
                    "text": "My favorite professor was Dr. Sarah Chen who taught AI and machine learning",
                    "memory_type": "episodic_memory",
                    "tags": [
                        "education",
                        "professor",
                        "ai",
                        "machine_learning",
                        "inspiration",
                    ],
                    "scope": "education",
                },
                {
                    "text": "I wrote my senior thesis on neural networks for natural language processing",
                    "memory_type": "semantic_memory",
                    "tags": ["education", "thesis", "neural_networks", "nlp"],
                    "scope": "education",
                },
                {
                    "text": "During college, I interned at Microsoft for two summers working on search algorithms",
                    "memory_type": "episodic_memory",
                    "tags": [
                        "education",
                        "internship",
                        "microsoft",
                        "search_algorithms",
                    ],
                    "scope": "career",
                },
            ],
            "career_progression": [
                {
                    "text": "My first job after college was at Amazon as a Software Development Engineer in 2012",
                    "memory_type": "semantic_memory",
                    "tags": ["career", "amazon", "software_engineer", "first_job"],
                    "scope": "career",
                },
                {
                    "text": "At Amazon, I worked on the recommendation engine team for 3 years",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "career",
                        "amazon",
                        "recommendation_engine",
                        "machine_learning",
                    ],
                    "scope": "career",
                },
                {
                    "text": "In 2015, I joined Google as a Senior Software Engineer on the Search Quality team",
                    "memory_type": "semantic_memory",
                    "tags": ["career", "google", "senior_engineer", "search_quality"],
                    "scope": "career",
                },
                {
                    "text": "At Google, I led the development of a new ranking algorithm that improved search relevance by 15%",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "career",
                        "google",
                        "achievement",
                        "algorithm",
                        "leadership",
                    ],
                    "scope": "career",
                },
                {
                    "text": "In 2020, I became a Tech Lead at Google, managing a team of 8 engineers",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "career",
                        "google",
                        "tech_lead",
                        "management",
                        "leadership",
                    ],
                    "scope": "career",
                },
            ],
            "relationships_social": [
                {
                    "text": "I met my wife Lisa at a tech conference in 2016, she was presenting on mobile app security",
                    "memory_type": "episodic_memory",
                    "tags": [
                        "relationship",
                        "wife",
                        "tech_conference",
                        "security",
                        "meeting",
                    ],
                    "scope": "personal",
                },
                {
                    "text": "Lisa and I got married in 2019 in a small ceremony in Napa Valley",
                    "memory_type": "episodic_memory",
                    "tags": ["relationship", "marriage", "wedding", "napa_valley"],
                    "scope": "personal",
                },
                {
                    "text": "My best friend from college, Alex, is now a product manager at Apple",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "friendship",
                        "college",
                        "alex",
                        "apple",
                        "product_manager",
                    ],
                    "scope": "social",
                },
                {
                    "text": "Alex and I still meet every month for basketball and coffee to catch up",
                    "memory_type": "episodic_memory",
                    "tags": ["friendship", "basketball", "coffee", "routine"],
                    "scope": "social",
                },
                {
                    "text": "We have a daughter named Maya who was born in 2021",
                    "memory_type": "semantic_memory",
                    "tags": ["family", "daughter", "maya", "parenthood"],
                    "scope": "family",
                },
            ],
            "hobbies_interests": [
                {
                    "text": "I've been playing guitar since high school and own a vintage Fender Stratocaster",
                    "memory_type": "semantic_memory",
                    "tags": ["hobby", "guitar", "music", "fender", "creative"],
                    "scope": "personal",
                },
                {
                    "text": "I love hiking and have completed sections of the Pacific Crest Trail",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "hobby",
                        "hiking",
                        "pacific_crest_trail",
                        "nature",
                        "adventure",
                    ],
                    "scope": "personal",
                },
                {
                    "text": "Photography became a passion after I took a course in 2018, I specialize in landscape photos",
                    "memory_type": "episodic_memory",
                    "tags": ["hobby", "photography", "landscape", "creative", "course"],
                    "scope": "personal",
                },
                {
                    "text": "I volunteer at a local coding bootcamp teaching Python to career changers",
                    "memory_type": "semantic_memory",
                    "tags": [
                        "volunteer",
                        "teaching",
                        "python",
                        "coding_bootcamp",
                        "giving_back",
                    ],
                    "scope": "social",
                },
            ],
            "recent_events": [
                {
                    "text": "Last month I attended the NIPS conference where I presented our latest search algorithm research",
                    "memory_type": "episodic_memory",
                    "tags": [
                        "recent",
                        "conference",
                        "nips",
                        "presentation",
                        "research",
                    ],
                    "scope": "career",
                },
                {
                    "text": "Yesterday Lisa and I discussed moving to a larger house now that Maya is growing up",
                    "memory_type": "episodic_memory",
                    "tags": ["recent", "family", "house", "discussion", "decision"],
                    "scope": "family",
                },
                {
                    "text": "This week I started mentoring a new engineer on my team who just graduated from Stanford",
                    "memory_type": "episodic_memory",
                    "tags": [
                        "recent",
                        "mentoring",
                        "engineer",
                        "stanford",
                        "leadership",
                    ],
                    "scope": "career",
                },
                {
                    "text": "I recently started learning Spanish because we're planning a family trip to Barcelona",
                    "memory_type": "episodic_memory",
                    "tags": ["recent", "language", "spanish", "travel", "barcelona"],
                    "scope": "personal",
                },
            ],
            "reflections_growth": [
                {
                    "text": "Looking back, my internship at Microsoft was crucial - it taught me how large-scale systems work",
                    "memory_type": "meta_memory",
                    "tags": [
                        "reflection",
                        "microsoft",
                        "learning",
                        "systems",
                        "growth",
                    ],
                    "scope": "career",
                },
                {
                    "text": "I realize that having Dr. Chen as a mentor shaped my entire approach to machine learning",
                    "memory_type": "meta_memory",
                    "tags": [
                        "reflection",
                        "mentor",
                        "influence",
                        "machine_learning",
                        "gratitude",
                    ],
                    "scope": "education",
                },
                {
                    "text": "Becoming a parent has made me think differently about work-life balance and priorities",
                    "memory_type": "meta_memory",
                    "tags": [
                        "reflection",
                        "parenthood",
                        "balance",
                        "priorities",
                        "growth",
                    ],
                    "scope": "personal",
                },
                {
                    "text": "I've learned that my technical curiosity from childhood directly led to my career success",
                    "memory_type": "meta_memory",
                    "tags": [
                        "reflection",
                        "childhood",
                        "curiosity",
                        "career",
                        "connection",
                    ],
                    "scope": "identity",
                },
                {
                    "text": "Teaching at the coding bootcamp has reminded me how much I enjoy explaining complex concepts",
                    "memory_type": "meta_memory",
                    "tags": [
                        "reflection",
                        "teaching",
                        "communication",
                        "passion",
                        "discovery",
                    ],
                    "scope": "personal",
                },
            ],
            "skills_expertise": [
                {
                    "text": "I'm expert-level in Python, Java, and C++, with deep knowledge of algorithms and data structures",
                    "memory_type": "procedural_memory",
                    "tags": ["skills", "programming", "algorithms", "expertise"],
                    "scope": "professional",
                },
                {
                    "text": "I have extensive experience with machine learning frameworks like TensorFlow and PyTorch",
                    "memory_type": "procedural_memory",
                    "tags": ["skills", "machine_learning", "tensorflow", "pytorch"],
                    "scope": "professional",
                },
                {
                    "text": "My leadership experience includes mentoring 20+ engineers and leading cross-functional projects",
                    "memory_type": "procedural_memory",
                    "tags": ["skills", "leadership", "mentoring", "project_management"],
                    "scope": "professional",
                },
            ],
            "future_aspirations": [
                {
                    "text": "I'm considering starting my own AI company focused on ethical machine learning",
                    "memory_type": "semantic_memory",
                    "tags": ["future", "entrepreneurship", "ai", "ethics", "ambition"],
                    "scope": "career",
                },
                {
                    "text": "Lisa and I want to travel to Japan next year to experience the culture and technology scene",
                    "memory_type": "semantic_memory",
                    "tags": ["future", "travel", "japan", "culture", "technology"],
                    "scope": "personal",
                },
                {
                    "text": "I hope to write a book about the intersection of technology and human creativity",
                    "memory_type": "semantic_memory",
                    "tags": ["future", "writing", "book", "technology", "creativity"],
                    "scope": "personal",
                },
            ],
        }

    async def create_memory_batch(
        self, memories: List[Dict[str, Any]], category: str
    ) -> int:
        """Create a batch of memories for a specific category"""
        created_count = 0

        for i, memory_data in enumerate(memories):
            payload = {
                "user_id": SAMPLE_USER_ID,
                "session_id": SAMPLE_SESSION_ID,
                **memory_data,
                "metadata": {
                    "category": category,
                    "sequence": i + 1,
                    "sample_data": True,
                    "created_date": datetime.now().isoformat(),
                },
            }

            async with self.session.post(
                f"{BASE_URL}/memories", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        memory_id = result["result"]["memory_id"]
                        self.created_memories.append(memory_id)
                        created_count += 1
                        print(f"   ✅ {memory_data['text'][:60]}...")
                    else:
                        print(f"   ❌ Failed: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")

        return created_count

    async def create_all_sample_memories(self):
        """Create all sample memories across categories"""
        print("🧠 Creating Comprehensive Sample Dataset...")
        print("=" * 70)

        sample_data = self.get_comprehensive_sample_data()
        total_created = 0

        for category, memories in sample_data.items():
            print(f"\n📂 Creating {category.replace('_', ' ').title()} memories:")
            created = await self.create_memory_batch(memories, category)
            total_created += created
            self.memory_categories[category] = created
            print(f"   📊 Created {created}/{len(memories)} memories")

        print(f"\n" + "=" * 70)
        print(f"✅ Sample Data Creation Complete!")
        print(f"📊 Total memories created: {total_created}")
        print(f"🗂️  Categories: {len(sample_data)}")

        # Show category breakdown
        print(f"\n📋 Memory Distribution:")
        for category, count in self.memory_categories.items():
            print(f"   {category.replace('_', ' ').title()}: {count} memories")

        return total_created

    async def verify_sample_data(self):
        """Verify the created sample data with test queries"""
        print(f"\n🔍 Verifying Sample Data with Test Queries...")

        test_queries = [
            "Tell me about my education",
            "How did I meet my wife?",
            "What's my relationship with my sister Emma?",
            "How has my career progressed over time?",
            "What are my hobbies and interests?",
        ]

        for query in test_queries:
            print(f"\n   ❓ {query}")

            payload = {
                "question": query,
                "user_id": SAMPLE_USER_ID,
                "session_id": SAMPLE_SESSION_ID,
                "max_memories": 5,
                "search_depth": "semantic",
            }

            async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        answer = result["result"]["answer"]
                        memories_used = result["result"]["memories_used"]
                        confidence = result["result"]["confidence"]

                        print(f"   💬 {answer[:100]}...")
                        print(
                            f"   📊 Used {memories_used} memories (confidence: {confidence:.2f})"
                        )
                    else:
                        print(f"   ❌ Query failed: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")


async def main():
    """Main sample data creation script"""
    print("🚀 Sample Data Creator for Multi-Hop Reasoning")
    print("Creating comprehensive, interconnected memories for testing")
    print()

    # Check server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/ping") as response:
                if response.status != 200:
                    print(
                        "❌ Server not running. Start with: poetry run uvicorn app.main:app --reload"
                    )
                    return
                print("✅ Server is running")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Create sample data
    async with SampleDataCreator() as creator:
        total_memories = await creator.create_all_sample_memories()

        if total_memories > 0:
            await creator.verify_sample_data()

            print(f"\n🎯 Ready for Multi-Hop Testing!")
            print(f"📋 Use test_multihop_reasoning.py to test complex queries")
            print(f"💡 Sample user ID: {SAMPLE_USER_ID}")
            print(f"🔍 Try questions like:")
            print(f"   • 'How do my childhood interests connect to my career success?'")
            print(f"   • 'What role has mentorship played throughout my life?'")
            print(
                f"   • 'How do my family relationships influence my professional decisions?'"
            )
        else:
            print("❌ No memories were created successfully")


if __name__ == "__main__":
    asyncio.run(main())
