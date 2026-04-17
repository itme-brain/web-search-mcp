FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastmcp httpx
COPY server.py .
CMD ["python", "server.py"]
