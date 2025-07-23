#!/usr/bin/env python3
"""
Print all loaded configuration for Context0 backend
"""

import json
import os
import sys
from typing import Any, Dict


def print_section(title: str, data: Any, indent: int = 0):
    """Print a configuration section with nice formatting"""
    prefix = "  " * indent
    print(f"{prefix}{'='*60}")
    print(f"{prefix}📋 {title}")
    print(f"{prefix}{'='*60}")

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: [{len(value)} items]")
                if value and len(value) < 10:  # Show small lists
                    for i, item in enumerate(value):
                        print(f"{prefix}  [{i}] {item}")
            else:
                # Mask sensitive values
                if any(
                    sensitive in key.lower()
                    for sensitive in ["key", "password", "secret", "token"]
                ):
                    if value and len(str(value)) > 4:
                        masked_value = "*" * (len(str(value)) - 4) + str(value)[-4:]
                    else:
                        masked_value = "****"
                    print(f"{prefix}{key}: {masked_value}")
                else:
                    print(f"{prefix}{key}: {value}")
    else:
        print(f"{prefix}{data}")
    print()


def print_dict(data: Dict[str, Any], indent: int = 0):
    """Print dictionary with proper indentation"""
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: [{len(value)} items]")
            if value and len(value) < 10:
                for i, item in enumerate(value):
                    print(f"{prefix}  [{i}] {item}")
        else:
            # Mask sensitive values
            if any(
                sensitive in key.lower()
                for sensitive in ["key", "password", "secret", "token"]
            ):
                if value and len(str(value)) > 4:
                    masked_value = "*" * (len(str(value)) - 4) + str(value)[-4:]
                else:
                    masked_value = "****"
                print(f"{prefix}{key}: {masked_value}")
            else:
                print(f"{prefix}{key}: {value}")


def main():
    print("🔧 Context0 Backend Configuration Inspector")
    print("=" * 80)

    # 1. Environment Variables
    print_section(
        "Environment Variables",
        {
            "AUTH_DISABLED": os.getenv("AUTH_DISABLED", "false"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "DEBUG": os.getenv("DEBUG", "false"),
            # Database URLs
            "MONGODB_URL": os.getenv("MONGODB_URL", "mongodb://localhost:27017"),
            "POSTGRES_URL": os.getenv("POSTGRES_URL", "not_set"),
            "TIMESCALEDB_URL": os.getenv("TIMESCALEDB_URL", "not_set"),
            # API Keys
            "CLERK_API_KEY": os.getenv("CLERK_API_KEY", "not_set"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", "not_set"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "not_set"),
            # Pinecone Config
            "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"),
            "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "memory-index"),
            # Neo4j Config
            "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "not_set"),
        },
    )

    # 2. Load and display memory system configuration
    try:
        sys.path.append("/Users/sunil28/PycharmProjects/context0-python-backend")
        from app.memory.config.yaml_config_loader import YamlConfigLoader

        print_section("Memory System Configuration", "Loading from YAML files...")

        config_loader = YamlConfigLoader()

        # Load different config types
        doc_store_config = config_loader.get_doc_store_config()
        print_section("Document Store Configuration", doc_store_config)

        vector_config = config_loader.get_vector_store_config()
        print_section("Vector Store Configuration", vector_config)

        graph_config = config_loader.get_graph_store_config()
        print_section("Graph Store Configuration", graph_config)

        llm_config = config_loader.get_llm_config()
        print_section("LLM Configuration", llm_config)

        embedder_config = config_loader.get_embedder_config()
        print_section("Embedder Configuration", embedder_config)

        system_config = config_loader.get_system_config()
        print_section("System Configuration", system_config)

    except Exception as e:
        print(f"❌ Error loading memory system config: {e}")

    # 3. Load app settings
    try:
        from app.settings import settings

        print_section(
            "App Settings",
            {
                "clerk_api_key": settings.clerk_api_key,
                "auth_disabled": settings.auth_disabled,
                # Add other settings fields as needed
            },
        )
    except Exception as e:
        print(f"❌ Error loading app settings: {e}")

    # 4. Python environment info
    print_section(
        "Python Environment",
        {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform,
            "current_directory": os.getcwd(),
        },
    )

    # 5. Check which packages are installed
    print_section("Key Package Availability", "Checking imports...")

    packages_to_check = [
        "pinecone",
        "openai",
        "motor",
        "pymongo",
        "neo4j",
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
    ]

    package_status = {}
    for package in packages_to_check:
        try:
            __import__(package)
            package_status[package] = "✅ Available"
        except ImportError as e:
            package_status[package] = f"❌ Missing: {e}"

    print_dict(package_status, 1)

    # 6. Configuration file locations
    config_files = [
        "/Users/sunil28/PycharmProjects/context0-python-backend/app/memory/config/default.yaml",
        "/Users/sunil28/PycharmProjects/context0-python-backend/app/memory/config/prod.yaml",
        "/Users/sunil28/PycharmProjects/context0-python-backend/.env",
        "/Users/sunil28/PycharmProjects/context0-python-backend/pyproject.toml",
    ]

    print_section("Configuration File Status", "Checking file existence...")
    for file_path in config_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} (not found)")

    print("\n" + "=" * 80)
    print("🎯 Configuration inspection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
