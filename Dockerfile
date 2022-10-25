FROM python:3.11-slim-buster as builder

# install system dependencies
RUN apt-get update \
  && apt-get -y install gcc \
  && apt-get clean

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# lint
COPY . /usr/src/app/
#RUN pip install black==20.8b1 flake8===3.8.4 isort==5.7.0
#RUN flake8 .
#RUN black --exclude=migrations .
#RUN isort .



FROM python:3.11-slim-buster

# create directory for the app user
RUN mkdir -p /home/app

# create the app user
RUN addgroup --system app && adduser --system --group app

# create the appropriate directories
ENV HOME=/home/app
RUN mkdir $HOME/src
WORKDIR $HOME/src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV ENVIRONMENT prod
ENV TESTING 0

# install system dependencies
RUN apt-get update \
  && apt-get -y install  gcc  \
  && apt-get clean

# install python dependencies
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*

# add app
COPY . .

# chown all the files to the app user
RUN chown -R app:app $HOME

# change to the app user
USER app

ENTRYPOINT ["python3"]
CMD ["preprocess.py"]