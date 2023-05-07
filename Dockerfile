FROM python:3.10
ADD ./jester/ /jester/
RUN pip install --upgrade pip
RUN pip3 install -r jester/setting/requirements.txt
CMD ["python", "jester/start.py"] 