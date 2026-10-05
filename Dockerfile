FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python-headless mediapipe scikit-learn pillow joblib numpy flask \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY common ./common
COPY web_app ./web_app
COPY baseline_a_landmarks ./baseline_a_landmarks
COPY baseline_b_cnn ./baseline_b_cnn
COPY robot ./robot
COPY models ./models

EXPOSE 5000
CMD ["python", "web_app/app.py"]
