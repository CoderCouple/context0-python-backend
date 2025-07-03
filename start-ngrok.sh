#!/bin/bash

# === CONFIG ===
PORT=8000                  # Local server port
NGROK_REGION=us            # Optional: set your preferred region
SERVER_CMD="uvicorn app.main:app --port $PORT"  # Replace with your dev server command
LOG_DIR="./logs"
NGROK_LOG="$LOG_DIR/ngrok.log"
NGROK_CONFIG="./ngrok.yml"

# === PREP ===
mkdir -p "$LOG_DIR"
echo "🧹 Cleaning up old logs..."
rm -f "$NGROK_LOG"

# === STEP 1: Start your local server ===
echo "🚀 Starting local server on port $PORT..."
$SERVER_CMD &  # Run in background
SERVER_PID=$!

# Give the server a second to boot
sleep 2

# === STEP 2: Start Ngrok ===
echo "🌐 Opening Ngrok tunnel to port $PORT..."
ngrok start --config="$NGROK_CONFIG" api-server --log=stdout > "$NGROK_LOG" &
#ngrok http --region=$NGROK_REGION $PORT --log=stdout > "$NGROK_LOG" &
NGROK_PID=$!

# === STEP 3: Open Ngrok dashboard
if ! command -v jq &> /dev/null; then
  echo "❌ jq is not installed. Please install it with 'brew install jq' or 'sudo apt install jq'"
  exit 1
fi

sleep 2
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')
echo "🌍 Ngrok Public URL: $PUBLIC_URL"
open http://localhost:4040


# === STEP 4: Tail logs
echo "📡 Tailing Ngrok logs:"
tail -f "$NGROK_LOG"

# === CLEANUP ON EXIT ===
trap "echo '🛑 Shutting down...'; kill $SERVER_PID $NGROK_PID" SIGINT
wait
