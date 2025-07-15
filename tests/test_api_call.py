#!/usr/bin/env python3
"""Test memory API with actual HTTP calls"""

import asyncio
import json

import httpx


async def test_memory_creation():
    """Test creating a memory via HTTP API"""

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Test health endpoint first
        print("Testing health endpoint...")
        try:
            health_response = await client.get(f"{base_url}/api/v1/health")
            print(f"Health status: {health_response.status_code}")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"Health data: {json.dumps(health_data, indent=2)}")
            else:
                print(f"Health response: {health_response.text}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return

        # Test memory creation
        print("\nTesting memory creation...")
        memory_data = {
            "user_id": "test-user-123",
            "session_id": "test-session-456",
            "text": "I learned how to use Python async/await patterns effectively today. This was particularly useful for handling database operations.",
            "tags": ["python", "programming", "learning"],
            "metadata": {"source": "api_test", "importance": "high"},
        }

        try:
            create_response = await client.post(
                f"{base_url}/api/v1/memories",
                json=memory_data,
                headers={"Content-Type": "application/json"},
            )

            print(f"Create status: {create_response.status_code}")
            if create_response.status_code == 200:
                create_data = create_response.json()
                print(f"Create response: {json.dumps(create_data, indent=2)}")

                if create_data.get("success") and create_data.get("result", {}).get(
                    "success"
                ):
                    memory_id = create_data.get("result", {}).get("memory_id")
                    print(f"✅ Memory created successfully! ID: {memory_id}")

                    # Test search to verify it was stored
                    print("\nTesting memory search...")
                    search_data = {
                        "user_id": "test-user-123",
                        "query": "Python async patterns",
                        "limit": 5,
                    }

                    search_response = await client.post(
                        f"{base_url}/api/v1/memories/search",
                        json=search_data,
                        headers={"Content-Type": "application/json"},
                    )

                    print(f"Search status: {search_response.status_code}")
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        print(f"Search results: {json.dumps(search_data, indent=2)}")

                        results = search_data.get("result", {}).get("results", [])
                        if results:
                            print(f"✅ Found {len(results)} memories in search!")
                        else:
                            print("❌ No memories found in search")
                    else:
                        print(f"Search failed: {search_response.text}")
                else:
                    print(f"❌ Memory creation failed: {create_data}")
            else:
                print(f"Create failed: {create_response.text}")

        except Exception as e:
            print(f"❌ Memory creation failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_memory_creation())
