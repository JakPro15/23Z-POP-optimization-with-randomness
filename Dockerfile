FROM python:3.11
WORKDIR /pop
ADD . /pop
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT cd /pop; bash
