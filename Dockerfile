FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

WORKDIR /app

COPY . .

RUN pip install -e .

ENTRYPOINT ["python", "run_example.py"]