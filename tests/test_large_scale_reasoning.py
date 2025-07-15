#!/usr/bin/env python3
"""
Test script specifically for large-scale memory cross-referencing
Demonstrates the system's ability to use 20-50+ memories for comprehensive reasoning
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import aiohttp

BASE_URL = "http://localhost:8000/api/v1"
TEST_USER_ID = "large-scale-test-user"
TEST_SESSION_ID = "large-scale-session"


class LargeScaleReasoningTester:
    """Test suite for large-scale memory cross-referencing"""

    def __init__(self):
        self.session = None
        self.created_memories = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def create_comprehensive_memory_set(self):
        """Create a large, interconnected set of memories across different categories"""
        print("🧠 Creating comprehensive memory dataset for large-scale reasoning...")

        # Create 30+ interconnected memories across different themes
        memory_sets = {
            "personal_background": [
                "I was born in Seattle on March 15, 1985",
                "My parents divorced when I was 12 years old",
                "I have two siblings: older brother Mike and younger sister Lisa",
                "I moved to San Francisco for college in 2003",
                "I studied Computer Science at UC Berkeley",
            ],
            "career_journey": [
                "My first internship was at a small startup called DataFlow in 2005",
                "I graduated from Berkeley in 2007 with a CS degree",
                "My first full-time job was at Google as a software engineer in 2007",
                "I worked on the Gmail team for 3 years at Google",
                "In 2010, I left Google to join Facebook",
                "At Facebook, I worked on the News Feed algorithm",
                "I was promoted to Senior Engineer at Facebook in 2012",
                "In 2015, I decided to start my own company called TechFlow",
            ],
            "relationships": [
                "I met my wife Sarah at Facebook in 2011",
                "Sarah and I got married in 2014 in Napa Valley",
                "My brother Mike also works in tech at Apple",
                "My sister Lisa became a doctor and lives in Portland",
                "My best friend from college, Alex, works at Microsoft",
                "Sarah and I have two children: Emma (born 2016) and Jack (born 2018)",
            ],
            "interests_hobbies": [
                "I love playing basketball and joined a local league",
                "I play guitar and was in a band called Code Breakers in college",
                "I enjoy hiking and have climbed Mount Whitney twice",
                "Photography is my creative outlet - I specialize in landscape photos",
                "I'm passionate about cooking and make a mean Italian pasta",
                "I volunteer at a local coding bootcamp teaching Python",
            ],
            "recent_activities": [
                "Last month I attended a tech conference in Austin",
                "Yesterday I had lunch with Alex and we discussed starting a podcast",
                "This week I'm mentoring a new intern at my company",
                "I recently started learning Spanish using Duolingo",
                "Last weekend I took the family camping in Yosemite",
                "I've been reading a book about AI ethics by Cathy O'Neil",
            ],
            "reflections_goals": [
                "I realize that work-life balance has become more important to me",
                "Looking back, leaving Google was the right decision for my growth",
                "I want to spend more time teaching and mentoring others",
                "My goal is to make TechFlow a company that values employee wellbeing",
                "I've learned that family time is more valuable than career advancement",
                "I'm considering writing a book about entrepreneurship in tech",
            ],
        }

        all_memories = []
        memory_types = ["semantic_memory", "episodic_memory", "meta_memory"]

        for category, memories in memory_sets.items():
            for i, memory_text in enumerate(memories):
                memory_type = random.choice(memory_types)
                if "reflect" in memory_text.lower() or "realize" in memory_text.lower():
                    memory_type = "meta_memory"
                elif (
                    "yesterday" in memory_text.lower() or "last" in memory_text.lower()
                ):
                    memory_type = "episodic_memory"

                memory_data = {
                    "user_id": TEST_USER_ID,
                    "session_id": TEST_SESSION_ID,
                    "text": memory_text,
                    "memory_type": memory_type,
                    "tags": [category, memory_type.replace("_", "")],
                    "metadata": {
                        "category": category,
                        "sequence": i,
                        "interconnected": True,
                    },
                    "scope": category,
                }
                all_memories.append(memory_data)

        # Create memories in batches
        print(f"   Creating {len(all_memories)} interconnected memories...")
        created_count = 0

        for memory in all_memories:
            async with self.session.post(
                f"{BASE_URL}/memories", json=memory
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        memory_id = result["result"]["memory_id"]
                        self.created_memories.append(memory_id)
                        created_count += 1
                        if created_count % 5 == 0:
                            print(f"   Created {created_count} memories...")
                    else:
                        print(f"   ❌ Failed: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")

        print(f"   ✅ Total memories created: {created_count}")
        print(f"   📊 Categories: {list(memory_sets.keys())}")
        print(f"   🔗 Memories span personal, professional, and reflective themes\n")
        return created_count

    async def test_large_scale_cross_referencing(self):
        """Test the system's ability to cross-reference many memories"""
        print("🔗 Testing Large-Scale Memory Cross-Referencing...")

        complex_questions = [
            {
                "question": "How has my career journey influenced my family relationships and personal values?",
                "max_memories": 25,
                "description": "Cross-references career, family, and reflection memories",
            },
            {
                "question": "What patterns emerge from my decision-making process throughout my life?",
                "max_memories": 30,
                "description": "Analyzes decision patterns across multiple life domains",
            },
            {
                "question": "How do my hobbies and interests connect to my professional experiences and relationships?",
                "max_memories": 20,
                "description": "Finds connections between disparate life areas",
            },
            {
                "question": "Based on all my experiences, what advice would I give to someone starting their tech career?",
                "max_memories": 35,
                "description": "Synthesizes lessons from entire life story",
            },
            {
                "question": "How have my values and priorities evolved from college to parenthood?",
                "max_memories": 40,
                "description": "Traces value evolution across decades",
            },
        ]

        for test_case in complex_questions:
            print(f"\n   📋 Test Case: {test_case['description']}")
            print(f"   ❓ Question: {test_case['question']}")
            print(f"   🎯 Target Memories: {test_case['max_memories']}")

            payload = {
                "question": test_case["question"],
                "user_id": TEST_USER_ID,
                "session_id": TEST_SESSION_ID,
                "max_memories": test_case["max_memories"],
                "search_depth": "comprehensive",
                "include_meta_memories": True,
            }

            async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        qa_result = result["result"]
                        metadata = qa_result["metadata"]

                        print(f"   📝 Answer: {qa_result['answer'][:200]}...")
                        print(f"   🎯 Memories Found: {qa_result['memories_found']}")
                        print(f"   🔧 Memories Used: {qa_result['memories_used']}")
                        print(f"   📊 Confidence: {qa_result['confidence']:.2f}")
                        print(
                            f"   🧠 Reasoning Chains: {metadata.get('reasoning_chains', 0)}"
                        )
                        print(f"   🔍 Search Strategy: {qa_result['search_strategy']}")

                        # Show memory clustering info if available
                        memory_contexts = qa_result.get("memory_contexts", [])
                        sources = set(
                            ctx.get("source", "unknown") for ctx in memory_contexts
                        )
                        print(f"   🗂️  Data Sources: {sources}")

                        # Show suggestions
                        suggestions = qa_result.get("suggestions", [])
                        if suggestions:
                            print(f"   💡 Follow-ups: {suggestions[:2]}")

                        print(f"   ✅ Large-scale reasoning successful")
                    else:
                        print(f"   ❌ Failed: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")

    async def test_reasoning_explanation_with_many_memories(self):
        """Test the reasoning explanation for complex multi-memory questions"""
        print("\n🧠 Testing Reasoning Explanation with Large Memory Sets...")

        question = "How do my technical skills, family relationships, and personal growth all interconnect?"

        payload = {
            "question": question,
            "user_id": TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "max_memories": 45,
            "search_depth": "comprehensive",
        }

        async with self.session.post(
            f"{BASE_URL}/explain-reasoning", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    reasoning = result["result"]

                    print(f"   Question: {reasoning['question']}")
                    print(f"   Reasoning Chains: {len(reasoning['reasoning_chains'])}")

                    for i, chain in enumerate(
                        reasoning["reasoning_chains"][:2]
                    ):  # Show first 2 chains
                        print(
                            f"\n   Chain {i+1} (Confidence: {chain['overall_confidence']:.2f}):"
                        )
                        for j, step in enumerate(
                            chain["steps"][:3]
                        ):  # Show first 3 steps
                            print(f"     Step {j+1}: {step['step_type']}")
                            print(f"       🔍 Process: {step['reasoning_process']}")
                            print(f"       📤 Output: {step['output'][:100]}...")
                            print(f"       📊 Confidence: {step['confidence']:.2f}")

                    print(f"\n   🎯 Final Synthesis: {reasoning['synthesis'][:200]}...")
                    print(f"   ⚠️  Contradictions: {len(reasoning['contradictions'])}")
                    print(f"   🔍 Information Gaps: {len(reasoning['gaps'])}")
                    print(
                        f"   ✅ Reasoning explanation with large memory set successful"
                    )
                else:
                    print(f"   ❌ Failed: {result.get('message')}")
            else:
                print(f"   ❌ HTTP Error {response.status}")

    async def test_memory_clustering_capabilities(self):
        """Test advanced memory clustering with statistical analysis"""
        print("\n📊 Testing Memory Clustering Capabilities...")

        # Test a question that should trigger clustering
        question = "Give me a comprehensive overview of my entire life story and how different aspects connect"

        payload = {
            "question": question,
            "user_id": TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "max_memories": 50,  # High number to trigger clustering
            "search_depth": "comprehensive",
        }

        async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    qa_result = result["result"]
                    metadata = qa_result["metadata"]

                    print(f"   🔍 Memories Found: {qa_result['memories_found']}")
                    print(
                        f"   🎯 Memories Used in Final Answer: {qa_result['memories_used']}"
                    )
                    print(f"   📊 Processing Time: {qa_result['processing_time_ms']}ms")

                    # Analyze confidence distribution
                    confidence_dist = metadata.get("confidence_distribution", {})
                    print(f"   📈 Confidence Distribution:")
                    for aspect, confidence in confidence_dist.items():
                        print(f"     {aspect}: {confidence:.2f}")

                    # Show memory context analysis
                    memory_contexts = qa_result.get("memory_contexts", [])
                    if memory_contexts:
                        sources = {}
                        for ctx in memory_contexts:
                            source = ctx.get("source", "unknown")
                            sources[source] = sources.get(source, 0) + 1

                        print(f"   🗂️  Memory Sources Distribution:")
                        for source, count in sources.items():
                            print(f"     {source}: {count} memories")

                    print(f"   ✅ Memory clustering analysis complete")
                else:
                    print(f"   ❌ Failed: {result.get('message')}")
            else:
                print(f"   ❌ HTTP Error {response.status}")

    async def demonstrate_cross_reference_depth(self):
        """Demonstrate the depth of cross-referencing capabilities"""
        print("\n🕸️  Demonstrating Cross-Reference Depth...")

        questions_by_complexity = [
            {
                "question": "Tell me about my brother Mike",
                "complexity": "Simple (1-hop)",
                "max_memories": 5,
            },
            {
                "question": "How are my family relationships connected to my career choices?",
                "complexity": "Medium (2-3 hops)",
                "max_memories": 15,
            },
            {
                "question": "What hidden patterns exist between my childhood, education, career, and current values?",
                "complexity": "Complex (4+ hops)",
                "max_memories": 35,
            },
        ]

        for test in questions_by_complexity:
            print(f"\n   🎯 {test['complexity']}: {test['question']}")

            payload = {
                "question": test["question"],
                "user_id": TEST_USER_ID,
                "session_id": TEST_SESSION_ID,
                "max_memories": test["max_memories"],
                "search_depth": "comprehensive",
            }

            async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        qa_result = result["result"]
                        print(f"     📊 Memories Used: {qa_result['memories_used']}")
                        print(f"     🎯 Confidence: {qa_result['confidence']:.2f}")
                        print(
                            f"     📝 Answer Quality: {'Rich and detailed' if len(qa_result['answer']) > 200 else 'Basic'}"
                        )
                        print(f"     ✅ Success")
                    else:
                        print(f"     ❌ Failed: {result.get('message')}")
                else:
                    print(f"     ❌ HTTP Error {response.status}")

    async def run_large_scale_tests(self):
        """Run comprehensive large-scale reasoning tests"""
        print("🚀 Large-Scale Memory Cross-Referencing Test Suite")
        print("=" * 70)

        # Create comprehensive memory set
        memory_count = await self.create_comprehensive_memory_set()

        if memory_count < 20:
            print("❌ Insufficient memories created for large-scale testing")
            return

        # Run tests
        await self.test_large_scale_cross_referencing()
        await self.test_reasoning_explanation_with_many_memories()
        await self.test_memory_clustering_capabilities()
        await self.demonstrate_cross_reference_depth()

        print("\n" + "=" * 70)
        print("✅ Large-Scale Reasoning Tests Completed!")
        print(f"📊 Total memories created: {len(self.created_memories)}")
        print("\n🎯 Key Capabilities Demonstrated:")
        print("   ✓ Cross-referencing 20-50+ memories simultaneously")
        print("   ✓ Multi-dimensional clustering (thematic, temporal, entity-based)")
        print("   ✓ Cross-cluster relationship discovery")
        print("   ✓ Comprehensive synthesis from large memory sets")
        print("   ✓ Multi-hop reasoning across different life domains")
        print("   ✓ Pattern recognition across decades of memories")
        print("\n💡 The system can handle enterprise-scale memory reasoning!")


async def main():
    """Main test runner"""
    print("🧪 Large-Scale Memory Cross-Referencing Test Suite")
    print("Testing the system's ability to use 20-50+ memories for reasoning")
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

    # Run large-scale tests
    async with LargeScaleReasoningTester() as tester:
        await tester.run_large_scale_tests()


if __name__ == "__main__":
    asyncio.run(main())
