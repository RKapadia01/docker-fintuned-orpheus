FROM python:3.10-slim

WORKDIR /
RUN pip install --no-cache-dir runpod
COPY rp_handler.py /

CMD ["python3", "-u", "rp_handler.py"]