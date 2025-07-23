#!/usr/bin/env python3
"""
Test Pinecone import and installation - Following mem0 pattern
"""

print("🔍 Testing Pinecone SDK installation (mem0 pattern)...")
print("=" * 60)

# Test 1: Test the exact import pattern from mem0
print("\n1. Testing mem0-style imports...")
try:
    from pinecone import Pinecone, PodSpec, ServerlessSpec

    print("✅ mem0-style imports successful")
    print(f"   - Pinecone: {Pinecone}")
    print(f"   - ServerlessSpec: {ServerlessSpec}")
    print(f"   - PodSpec: {PodSpec}")
except ImportError as e:
    print(f"❌ mem0-style import failed: {e}")
    print(
        "💡 Install with: poetry add 'pinecone[asyncio]' or pip install 'pinecone[asyncio]'"
    )
    exit(1)

# Test 2: Test our store import
print("\n2. Testing our PineconeVectorStore import...")
try:
    from app.memory.stores.pinecone_store import PineconeVectorStore

    print("✅ PineconeVectorStore imported successfully")
except Exception as e:
    print(f"❌ Store import failed: {e}")

# Test 3: Test environment variables
print("\n3. Testing environment variables...")
import os

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

if api_key:
    print(
        f"✅ PINECONE_API_KEY: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '****'}"
    )

    # Test 4: Test actual Pinecone client creation
    print("\n4. Testing Pinecone client creation...")
    try:
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client created successfully")

        # Test listing indexes (this will verify API connectivity)
        try:
            indexes = pc.list_indexes()
            print(f"✅ Connected to Pinecone - Found {len(indexes)} indexes")
            for idx in indexes:
                print(f"   📋 Index: {idx.name}")
        except Exception as e:
            print(f"⚠️  API connection test failed (but client creation worked): {e}")

    except Exception as e:
        print(f"❌ Pinecone client creation failed: {e}")
else:
    print("❌ PINECONE_API_KEY not set")
    print("💡 Set it in your .env file or export PINECONE_API_KEY=your_key")

print(f"✅ PINECONE_ENVIRONMENT: {env}")

# Test 5: Test ServerlessSpec creation
print("\n5. Testing ServerlessSpec creation...")
try:
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    print(f"✅ ServerlessSpec created: {spec}")
except Exception as e:
    print(f"❌ ServerlessSpec creation failed: {e}")

# Test 6: Version and package info
print("\n6. Package information...")
try:
    import pinecone

    if hasattr(pinecone, "__version__"):
        print(f"📦 Pinecone version: {pinecone.__version__}")
    else:
        print("📦 Version: Not available")

    # Check if asyncio extras are available
    try:
        import aiohttp

        print("✅ aiohttp available (asyncio extras installed)")
    except ImportError:
        print("⚠️  aiohttp not available (asyncio extras might not be installed)")

    try:
        import grpc

        print("✅ grpc available (gRPC extras installed)")
    except ImportError:
        print("ℹ️  grpc not available (gRPC extras not installed - this is OK)")

except Exception as e:
    print(f"❌ Package info error: {e}")

print("\n" + "=" * 60)
print("🎯 Summary:")
print("If all tests passed, your Pinecone SDK is properly installed!")
print("If there were issues:")
print("  1. Run: poetry install")
print("  2. Or: poetry add 'pinecone[asyncio]'")
print("  3. Make sure PINECONE_API_KEY is set in your .env file")
print("=" * 60)
