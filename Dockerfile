FROM python:3.8.18-slim

# install project requirements 
COPY ./requirements.txt ./docker/
RUN pip install --no-cache-dir -r ./docker/requirements.txt 

#first copy things that dont change often
#than you can copy things that change a lot
COPY ./data/ ./docker/data
COPY ./src/ ./docker/src

WORKDIR /docker/src/fastAPI

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
