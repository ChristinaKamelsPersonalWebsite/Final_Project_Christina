#!/bin/sh
# ollama-init.sh
# Waits for Ollama to be ready then pulls qwen2.5:7b if not already present.
# Runs as an init container before agent-system-a starts.

echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done

echo "Ollama is ready. Checking for qwen2.5:1.5b..."

# Check if model already exists
if ollama list | grep -q "qwen2.5:1.5b"; then
  echo "qwen2.5:1.5b already present. Skipping pull."
else
  echo "Pulling qwen2.5:1.5b — this may take a few minutes on first run..."
  ollama pull qwen2.5:1.5b
  echo "Pull complete."
fi

echo "Ollama init done."

