FROM python:3.12.8-slim-bullseye

COPY . /usr/app/
WORKDIR /usr/app/

#now set some env var
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SURPRISE_DATA_FOLDER=/tmp/surprise_data


# below define the working directory

# install the dependencies for surprise and others 
RUN apt-get update && apt-get install -y build-essential libatlas-base-dev && rm -rf /var/lib/apt/list/*
RUN pip install --upgrade pip

RUN pip install gradio==5.14.0
RUN pip install scikit-learn==1.5.2
RUN pip install scikit-surprise==1.1.4
RUN pip install pandas==2.2.3
RUN pip install numpy==1.26.4
RUN pip install joblib==1.4.2


EXPOSE 7860
CMD ["python", "app.py"]