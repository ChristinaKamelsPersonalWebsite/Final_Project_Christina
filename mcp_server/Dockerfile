FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY mcp_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy full project (needed for package imports)
COPY . /app

EXPOSE 8002

CMD ["python", "-m", "mcp_server.app.server"]