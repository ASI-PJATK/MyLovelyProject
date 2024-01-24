FROM python:3.8.18-slim

# install project requirements 
COPY /src/requirements.txt /docker/
RUN pip install --no-cache-dir -r /docker/requirements.txt 
RUN pip install fastapi
RUN pip install uvicorn

#first copy things that dont change often
#than you can copy things that change a lot
COPY /data/ /docker/data
COPY /src/ /docker/src

WORKDIR /docker/src/fastAPI

CMD ["uvicorn", "main:app"]