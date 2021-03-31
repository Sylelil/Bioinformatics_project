FROM python:3.6-stretch
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

RUN python3 --version
RUN pip3 --version

WORKDIR /usr/bioinf
COPY requirements.txt .
COPY datasets .

RUN pip install -r requirements.txt

RUN apt-get install -y openslide-tools && apt-get install -y python-openslide

COPY src src 

COPY datasets ./datasets

RUN mkdir extracted_features && mkdir extracted_features/images && mkdir extracted_features/images/numpy_normal && mkdir extracted_features/images/numpy_tumor


# Eseguiamo il primo script
CMD ["python3", "src/images_feature_extraction.py"]


