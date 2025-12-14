FROM python:3.12
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "lahmajo.api.routes:app", "--host", "0.0.0.0", "--port", "8080"]