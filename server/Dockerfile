FROM python:3.13-alpine
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ultralytics opencv-python face_recognition requests pyzmq aiohttp aiortc numpy pyyaml tqdm
EXPOSE 5555
EXPOSE 5556
EXPOSE 8080
CMD ["python", "-m", "server.security_server"]
