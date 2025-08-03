"""
context0_app.py
---
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.common.error.error_handlers import register_error_handlers
from app.common.logging.logging import setup_logging
from app.common.startup import app_start_time
from app.middleware.auth_middleware import AuthLoggingMiddleware
from app.settings import settings
from app.warning import print_boxed_auth_warning

# https://www.reddit.com/r/Python/comments/1dkrfgh/open_source_python_projects_with_good_software/
# https://github.com/blakeblackshear/frigate/blob/dev/frigate/config/auth.py
# https://github.com/polarsource/polar/blob/main/server/polar/api.py
# https://github.com/neuml/txtai/blob/master/src/python/txtai/app/base.py
# https://github.com/praw-dev/praw/blob/main/praw/config.py
# https://github.com/pallets/flask/blob/main/src/flask/ctx.py

# Set up Logging
logger = setup_logging()

if settings.auth_disabled:
    print_boxed_auth_warning()

# app_start_time is now imported from app.common.startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting application...")

    # Initialize MongoDB connection
    from app.db.mongodb import (
        connect_to_mongodb,
        close_mongodb_connection,
        get_database,
    )

    await connect_to_mongodb()

    # Create indexes for performance
    from app.db.indexes import create_indexes

    db = await get_database()
    await create_indexes(db)
    logger.info("MongoDB indexes created")

    # Initialize Memory System
    logger.info("Initializing Memory System...")
    from app.memory.config.context_zero_config import ContextZeroConfig
    from app.memory.engine.memory_engine import MemoryEngine

    # Initialize memory engine
    memory_config = ContextZeroConfig()
    memory_engine = MemoryEngine(memory_config)
    await memory_engine.initialize()

    # Store in app state for access in routes
    app.state.memory_engine = memory_engine

    logger.info("Memory System initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down application...")

    # Close MongoDB connection
    await close_mongodb_connection()

    # Shutdown Memory System
    if hasattr(app.state, "memory_engine"):
        await app.state.memory_engine.close()

    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ContextZero Memory System",
    description="High-performance memory management system with multi-store architecture",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)

# Configure security scheme for Swagger UI
security = HTTPBearer()

# Add security scheme to OpenAPI schema
from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ContextZero Memory System",
        version="1.0.0",
        description="High-performance memory management system with multi-store architecture",
        routes=app.routes,
    )

    # Add servers for base URL selection
    # Default server based on environment
    default_servers = []

    if settings.app_env == "production":
        default_servers.append(
            {
                "url": "https://api.context0.com",
                "description": "Production server (default)",
            }
        )
    else:
        default_servers.append(
            {
                "url": "http://localhost:8000",
                "description": "Local development server (default)",
            }
        )

    # Additional server options
    additional_servers = [
        {"url": "http://localhost:8000", "description": "Local development server"},
        {
            "url": "http://127.0.0.1:8000",
            "description": "Local development server (IP)",
        },
        {"url": "https://api.context0.com", "description": "Production server"},
        {"url": "https://staging-api.context0.com", "description": "Staging server"},
        {
            "url": "{customUrl}",
            "description": "Custom server URL",
            "variables": {
                "customUrl": {
                    "default": "http://localhost:8000",
                    "description": "Enter your custom API URL",
                }
            },
        },
    ]

    # Combine servers, avoiding duplicates
    openapi_schema["servers"] = default_servers
    for server in additional_servers:
        if server["url"] not in [s["url"] for s in openapi_schema["servers"]]:
            openapi_schema["servers"].append(server)

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your Clerk JWT token",
        }
    }

    # Apply security globally
    openapi_schema["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Add middleware
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # TODO: Replace "*" with your frontend domain like
    #  ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# add API router
app.include_router(api_router)

# Add Middleware
app.add_middleware(AuthLoggingMiddleware)

# Add Performance Monitoring Middleware
from app.common.monitoring.performance import PerformanceMiddleware

app.add_middleware(PerformanceMiddleware)

# Register error handlers
register_error_handlers(app)
