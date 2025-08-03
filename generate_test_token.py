#!/usr/bin/env python3
"""
Generate a test Clerk JWT token for API testing
"""
import jwt
import time
from datetime import datetime, timedelta
import json


def generate_test_token():
    """Generate a test JWT token that matches Clerk's format"""

    # Test user information
    user_id = "user_2xy5bIyLFPhOjYUsbJgsUpbjZxL"
    email = "test@example.com"

    # Current time
    now = int(time.time())

    # Token payload (similar to Clerk's format)
    payload = {
        "sub": user_id,
        "email": email,
        "iss": "https://clerk.example.com",  # Issuer
        "aud": "your-audience",  # Audience
        "iat": now,  # Issued at
        "exp": now + 3600,  # Expires in 1 hour
        "azp": "your-authorized-party",
        "session_id": "sess_test123",
        "jti": "jti_test123",
    }

    # Secret key (for testing only - in production this would be Clerk's key)
    secret = "your-test-secret-key-for-development-only"

    # Generate JWT
    token = jwt.encode(payload, secret, algorithm="HS256")

    print("=== Test Clerk JWT Token ===")
    print(f"Token: {token}")
    print(f"\nUser ID: {user_id}")
    print(f"Email: {email}")
    print(f"Expires: {datetime.fromtimestamp(payload['exp'])}")
    print(f"\nPayload: {json.dumps(payload, indent=2)}")
    print("\n=== Usage ===")
    print("In Swagger UI:")
    print("1. Click 'Authorize' button")
    print("2. Enter the token above in the 'Value' field")
    print("3. Click 'Authorize'")
    print("\nIn curl:")
    print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/api/v1/...')

    return token


if __name__ == "__main__":
    generate_test_token()
