#!/usr/bin/env python3
"""
Simple demo script for the Multi-Hop Q&A System
Shows basic usage and capabilities
"""

import asyncio
import json
from typing import Any, Dict

import aiohttp

BASE_URL = "http://localhost:8000/api/v1"


async def create_demo_memories():
    """Create some demo memories for testing"""
    print("🧠 Creating demo memories...")

    memories = [
        {
            "user_id": "demo-user",
            "session_id": "demo-session",
            "text": "I am born on 28 july",
            "memory_type": "semantic_memory",
            "tags": ["personal", "birth"],
            "metadata": {},
            "scope": "personal",
        },
        {
            "user_id": "demo-user",
            "session_id": "demo-session",
            "text": "My friend Sarah works at Microsoft as a software engineer",
            "memory_type": "semantic_memory",
            "tags": ["friend", "work", "microsoft"],
            "metadata": {},
            "scope": "social",
        },
        {
            "user_id": "demo-user",
            "session_id": "demo-session",
            "text": "Yesterday I had coffee with Sarah and she told me about her new project",
            "memory_type": "episodic_memory",
            "tags": ["friend", "coffee", "conversation"],
            "metadata": {},
            "scope": "recent",
        },
    ]

    async with aiohttp.ClientSession() as session:
        created_count = 0
        for memory in memories:
            async with session.post(f"{BASE_URL}/memories", json=memory) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        created_count += 1
                        print(f"   ✅ Created: {memory['text'][:50]}...")
                    else:
                        print(f"   ❌ Failed: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")

        print(f"   📊 Created {created_count} memories\n")


async def ask_questions():
    """Ask some demo questions"""
    print("❓ Asking questions...")

    questions = [
        "When was I born?",
        "Tell me about Sarah",
        "What do I know about Microsoft?",
        "How are Sarah and Microsoft connected in my memories?",
    ]

    async with aiohttp.ClientSession() as session:
        for question in questions:
            print(f"\n   Q: {question}")

            payload = {
                "question": question,
                "user_id": "demo-user",
                "session_id": "demo-session",
                "max_memories": 5,
                "search_depth": "comprehensive",
            }

            async with session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        answer_data = result["result"]
                        print(f"   A: {answer_data['answer']}")
                        print(f"   Confidence: {answer_data['confidence']:.2f}")
                        print(f"   Memories used: {answer_data['memories_used']}")

                        if answer_data.get("suggestions"):
                            print(f"   Suggestions: {answer_data['suggestions']}")
                    else:
                        print(f"   ❌ Failed: {result.get('message')}")
                else:
                    print(f"   ❌ HTTP Error {response.status}")


async def main():
    """Main demo"""
    print("🚀 Multi-Hop Q&A System Demo")
    print("=" * 50)

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

    # Run demo
    await create_demo_memories()
    await ask_questions()

    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\n🎯 Try these advanced questions:")
    print(
        "   • 'How do my relationships influence my understanding of technology companies?'"
    )
    print("   • 'What patterns do you see in my social interactions?'")
    print(
        "   • 'Based on my memories, what advice would you give me about career networking?'"
    )


if __name__ == "__main__":
    asyncio.run(main())
