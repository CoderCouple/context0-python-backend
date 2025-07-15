"""
context0_app.py
---
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
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
    logger.info("Shutting down Memory System...")
    if hasattr(app.state, "memory_engine"):
        await app.state.memory_engine.close()
    logger.info("Memory System shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ContextZero Memory System",
    description="High-performance memory management system with multi-store architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # TODO: Replace "*" with your frontend domain like
    #  ["http://localhost:3000", "https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# add API router
app.include_router(api_router)

# Add Middleware
app.add_middleware(AuthLoggingMiddleware)

# Register error handlers
register_error_handlers(app)
