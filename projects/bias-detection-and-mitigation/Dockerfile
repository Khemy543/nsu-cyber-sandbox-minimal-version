FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY projects/bias-detection-and-mitigation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "--version"]