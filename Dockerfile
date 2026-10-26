FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY researchops ./researchops
COPY dashboard.py .
COPY .streamlit ./.streamlit

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501

CMD ["uvicorn", "researchops.api:app", "--host", "0.0.0.0", "--port", "8000"]