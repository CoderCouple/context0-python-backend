"""
context0_app.py
---
"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.common.error.error_handlers import register_error_handlers
from app.common.logging import setup_logging
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
setup_logging()

if settings.auth_disabled:
    print_boxed_auth_warning()


app = FastAPI()

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

# add API router
app.include_router(api_router)

# Add Middleware
app.add_middleware(AuthLoggingMiddleware)

# Register error handlers
register_error_handlers(app)
