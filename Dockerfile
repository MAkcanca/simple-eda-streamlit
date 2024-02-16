FROM python:3.11
WORKDIR /app
COPY . /app

RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["streamlit", "run", "app.py"]