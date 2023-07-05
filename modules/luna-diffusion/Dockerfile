FROM stablecog/cuda-torch:11.8.0-2.0.1-cudnn8-devel-ubuntu22.04

ADD . .

RUN apt-get update && apt-get -y install git python3 python3-pip
RUN pip3 install -r requirements.txt --no-cache-dir
RUN python3 package/download.py

CMD ["python3", "main.py"]