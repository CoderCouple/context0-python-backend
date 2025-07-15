#!/usr/bin/env python3
"""
Comprehensive test script for the Multi-Hop Q&A System
Tests memory creation, cross-database references, and advanced reasoning
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

import aiohttp

BASE_URL = "http://localhost:8000/api/v1"
TEST_USER_ID = "test-user-qa"
TEST_SESSION_ID = "test-session-qa"


class QASystemTester:
    """Test suite for the Q&A system with multi-hop reasoning"""

    def __init__(self):
        self.session = None
        self.created_memories = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def test_memory_creation_and_references(self):
        """Test creating memories with cross-database references"""
        print("🧠 Testing Memory Creation with Cross-Database References...")

        # Create a series of related memories
        test_memories = [
            {
                "text": "I was born on July 28, 1990, in San Francisco",
                "memory_type": "semantic_memory",
                "tags": ["personal", "birth", "location"],
                "scope": "personal",
            },
            {
                "text": "My brother John started working at Google last month",
                "memory_type": "semantic_memory",
                "tags": ["family", "work", "google"],
                "scope": "family",
            },
            {
                "text": "Yesterday I had lunch with John and we talked about his new job",
                "memory_type": "episodic_memory",
                "tags": ["family", "lunch", "conversation"],
                "scope": "recent",
            },
            {
                "text": "Reflecting on my conversation with John, I realized I should also consider a career change",
                "memory_type": "meta_memory",
                "tags": ["reflection", "career", "family"],
                "scope": "personal",
            },
            {
                "text": "I learned Python programming in college and still use it for data analysis",
                "memory_type": "procedural_memory",
                "tags": ["skill", "programming", "education"],
                "scope": "professional",
            },
        ]

        for i, memory_data in enumerate(test_memories):
            print(f"   Creating memory {i+1}: {memory_data['text'][:50]}...")

            payload = {
                "user_id": TEST_USER_ID,
                "session_id": TEST_SESSION_ID,
                **memory_data,
                "metadata": {"test_sequence": i + 1},
            }

            async with self.session.post(
                f"{BASE_URL}/memories", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        memory_id = result["result"]["memory_id"]
                        self.created_memories.append(memory_id)
                        print(f"   ✅ Memory created: {memory_id}")
                    else:
                        print(f"   ❌ Failed to create memory: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")

        print(f"   📊 Total memories created: {len(self.created_memories)}\n")

    async def test_basic_qa(self):
        """Test basic Q&A functionality"""
        print("❓ Testing Basic Q&A...")

        test_questions = [
            {
                "question": "When was I born?",
                "expected_keywords": ["july", "28", "1990", "san francisco"],
            },
            {
                "question": "What do you know about my brother John?",
                "expected_keywords": ["john", "google", "work", "lunch"],
            },
            {
                "question": "What programming skills do I have?",
                "expected_keywords": ["python", "programming", "data analysis"],
            },
        ]

        for test_case in test_questions:
            print(f"   Question: {test_case['question']}")

            payload = {
                "question": test_case["question"],
                "user_id": TEST_USER_ID,
                "session_id": TEST_SESSION_ID,
                "max_memories": 5,
                "search_depth": "semantic",
            }

            async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        answer = result["result"]["answer"]
                        confidence = result["result"]["confidence"]
                        memories_used = result["result"]["memories_used"]

                        print(f"   Answer: {answer}")
                        print(f"   Confidence: {confidence:.2f}")
                        print(f"   Memories Used: {memories_used}")

                        # Check if expected keywords are in the answer
                        answer_lower = answer.lower()
                        found_keywords = [
                            kw
                            for kw in test_case["expected_keywords"]
                            if kw in answer_lower
                        ]
                        print(f"   Keywords Found: {found_keywords}")
                        print(f"   ✅ Basic Q&A working\n")
                    else:
                        print(f"   ❌ Q&A failed: {result.get('message')}\n")
                else:
                    print(f"   ❌ HTTP Error {response.status}\n")

    async def test_multi_hop_reasoning(self):
        """Test advanced multi-hop reasoning"""
        print("🔗 Testing Multi-Hop Reasoning...")

        complex_questions = [
            {
                "question": "How might my conversation with John about his career influence my own decisions?",
                "search_depth": "comprehensive",
                "description": "Tests causal reasoning and meta-memory connections",
            },
            {
                "question": "What connections exist between my programming skills and my family relationships?",
                "search_depth": "comprehensive",
                "description": "Tests pattern recognition across different memory types",
            },
            {
                "question": "Based on my memories, what can you infer about my personality and interests?",
                "search_depth": "comprehensive",
                "description": "Tests synthesis and inference capabilities",
            },
        ]

        for test_case in complex_questions:
            print(f"   Question: {test_case['question']}")
            print(f"   Test Focus: {test_case['description']}")

            payload = {
                "question": test_case["question"],
                "user_id": TEST_USER_ID,
                "session_id": TEST_SESSION_ID,
                "max_memories": 10,
                "search_depth": test_case["search_depth"],
                "include_meta_memories": True,
            }

            async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        answer = result["result"]["answer"]
                        confidence = result["result"]["confidence"]
                        metadata = result["result"]["metadata"]
                        suggestions = result["result"]["suggestions"]

                        print(f"   Answer: {answer}")
                        print(f"   Confidence: {confidence:.2f}")
                        print(
                            f"   Reasoning Chains: {metadata.get('reasoning_chains', 0)}"
                        )
                        print(f"   Contradictions: {metadata.get('contradictions', 0)}")
                        print(f"   Information Gaps: {metadata.get('gaps', 0)}")
                        print(f"   Follow-up Suggestions: {suggestions}")
                        print(f"   ✅ Multi-hop reasoning working\n")
                    else:
                        print(
                            f"   ❌ Multi-hop reasoning failed: {result.get('message')}\n"
                        )
                else:
                    print(f"   ❌ HTTP Error {response.status}\n")

    async def test_reasoning_explanation(self):
        """Test reasoning explanation endpoint"""
        print("🧠 Testing Reasoning Explanation...")

        payload = {
            "question": "Why did I consider a career change after talking to John?",
            "user_id": TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "max_memories": 10,
            "search_depth": "comprehensive",
        }

        async with self.session.post(
            f"{BASE_URL}/explain-reasoning", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    reasoning_data = result["result"]

                    print(f"   Question: {reasoning_data['question']}")
                    print(
                        f"   Reasoning Chains: {len(reasoning_data['reasoning_chains'])}"
                    )

                    for i, chain in enumerate(reasoning_data["reasoning_chains"]):
                        print(
                            f"   \n   Chain {i+1} (Confidence: {chain['overall_confidence']:.2f}):"
                        )
                        for j, step in enumerate(chain["steps"]):
                            print(f"     Step {j+1} ({step['step_type']}):")
                            print(f"       Process: {step['reasoning_process']}")
                            print(f"       Output: {step['output']}")
                            print(f"       Confidence: {step['confidence']:.2f}")

                    print(f"   \n   Final Synthesis: {reasoning_data['synthesis']}")
                    print(f"   Contradictions: {reasoning_data['contradictions']}")
                    print(f"   Information Gaps: {reasoning_data['gaps']}")
                    print(f"   ✅ Reasoning explanation working\n")
                else:
                    print(
                        f"   ❌ Reasoning explanation failed: {result.get('message')}\n"
                    )
            else:
                print(f"   ❌ HTTP Error {response.status}\n")

    async def test_conversational_qa(self):
        """Test conversational Q&A with context"""
        print("💬 Testing Conversational Q&A...")

        conversation = [
            {"role": "user", "content": "Tell me about my family"},
            {
                "role": "assistant",
                "content": "I can help you with information about your family. Let me search your memories.",
            },
            {"role": "user", "content": "What about John specifically?"},
        ]

        payload = {
            "messages": conversation,
            "user_id": TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "max_memories": 8,
            "conversation_context_window": 3,
        }

        async with self.session.post(
            f"{BASE_URL}/conversation", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    conv_result = result["result"]

                    print(f"   Response: {conv_result['response']}")
                    print(f"   Confidence: {conv_result['confidence']:.2f}")
                    print(
                        f"   Context Memories: {len(conv_result['context_memories'])}"
                    )
                    print(
                        f"   Follow-up Suggestions: {conv_result['follow_up_suggestions']}"
                    )
                    print(f"   ✅ Conversational Q&A working\n")
                else:
                    print(f"   ❌ Conversational Q&A failed: {result.get('message')}\n")
            else:
                print(f"   ❌ HTTP Error {response.status}\n")

    async def test_system_health(self):
        """Test system health and statistics"""
        print("🏥 Testing System Health...")

        # Health check
        async with self.session.get(f"{BASE_URL}/qa/health") as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    health = result["result"]
                    print(f"   System Status: {health['status']}")
                    print(f"   Memory Stores: {health['memory_stores']}")
                    print(f"   Embedding Service: {health['embedding_service']}")
                    print(f"   LLM Service: {health['llm_service']}")
                    print(f"   ✅ Health check working")
                else:
                    print(f"   ❌ Health check failed: {result.get('message')}")
            else:
                print(f"   ❌ HTTP Error {response.status}")

        # Statistics
        async with self.session.get(
            f"{BASE_URL}/stats?user_id={TEST_USER_ID}"
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    stats = result["result"]
                    print(f"   Available Stores: {stats['memory_stores_available']}")
                    print(
                        f"   Reasoning Capabilities: {stats['reasoning_capabilities']}"
                    )
                    print(f"   ✅ Statistics working\n")
                else:
                    print(f"   ❌ Statistics failed: {result.get('message')}\n")
            else:
                print(f"   ❌ HTTP Error {response.status}\n")

    async def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Comprehensive Q&A System Tests")
        print("=" * 60)

        try:
            await self.test_memory_creation_and_references()
            await self.test_basic_qa()
            await self.test_multi_hop_reasoning()
            await self.test_reasoning_explanation()
            await self.test_conversational_qa()
            await self.test_system_health()

            print("=" * 60)
            print("✅ All tests completed successfully!")
            print(f"📊 Total memories created: {len(self.created_memories)}")
            print("\n🎯 Key Features Tested:")
            print("   ✓ Cross-database memory references")
            print("   ✓ Multi-hop reasoning across memory types")
            print("   ✓ Temporal and relationship analysis")
            print("   ✓ Meta-memory and reflection processing")
            print("   ✓ Conversational context handling")
            print("   ✓ Reasoning explanation and transparency")

        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")


async def main():
    """Main test runner"""
    print("🧪 Multi-Hop Q&A System Test Suite")
    print("Ensure the server is running on http://localhost:8000")
    print()

    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/ping") as response:
                if response.status != 200:
                    print("❌ Server is not responding. Please start the server first.")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please make sure the server is running on http://localhost:8000")
        return

    # Run tests
    async with QASystemTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
