FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/cache/apt/archives /var/lib/apt/lists
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir wheel && pip install --no-cache-dir -r requirements.txt
RUN sed -i '599s|min_length=6|min_length=0|' /usr/local/lib/python3.10/site-packages/nostril/nonsense_detector.py
COPY categorizer.py classifier_prompt.txt ./
CMD [ "uvicorn", "categorizer:app", "--port", "80", "--host", "0.0.0.0" ]
