FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install cloudml-hypertune scikit-learn==0.20.4 pandas==0.24.2 gcsfs fsspec
WORKDIR /app
COPY task.py .

ENTRYPOINT ["python", "task.py"]
