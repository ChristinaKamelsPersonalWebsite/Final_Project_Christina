FROM python:3.11-slim-bookworm

WORKDIR /app

COPY agent_system_a/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 300 -r /app/requirements.txt
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "agent_system_a.app.main:app", "--host", "0.0.0.0", "--port", "8000"]