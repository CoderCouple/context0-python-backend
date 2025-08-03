# How to Obtain a Clerk Authentication Token

## Method 1: From Clerk Dashboard (Development Token)
1. Log in to your Clerk Dashboard at https://dashboard.clerk.com
2. Select your application
3. Go to "API Keys" section
4. Copy your Frontend API key or Backend API key as needed

## Method 2: From Your Frontend Application (User Token)
1. After a user logs in through Clerk in your frontend
2. Access the token using Clerk's SDK:

```javascript
// Using Clerk React
import { useAuth } from "@clerk/nextjs";

const { getToken } = useAuth();
const token = await getToken();
```

## Method 3: Using Clerk CLI (Development)
```bash
# Install Clerk CLI
npm install -g @clerk/cli

# Authenticate
clerk login

# Get development token
clerk token
```

## Method 4: Generate Test Token Programmatically
Run the existing test token generator:
```bash
python generate_test_token.py
```

This will create a token for the test user configured in your environment.

## Using the Token with Swagger UI
1. Open Swagger UI at http://localhost:8000/docs
2. Click the "Authorize" button (lock icon)
3. Enter your token in the "Value" field
4. Click "Authorize"
5. Now all API requests will include the Bearer token

## Token Format
The token should be used as a Bearer token in the Authorization header:
```
Authorization: Bearer <your-clerk-jwt-token>
```