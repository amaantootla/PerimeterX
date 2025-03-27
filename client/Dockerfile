FROM python:3.13-alpine
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ultralytics opencv-python face_recognition requests pyzmq aiohttp aiortc numpy pyyaml tqdm
EXPOSE 8080
ENV DEVICE_ID=client_01
CMD ["python", "-m", "client.security_client"]
