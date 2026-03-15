FROM python:3.11-slim

WORKDIR /app

# Install security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
CMD ["python", "src/random_search.py"]