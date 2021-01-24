# set base image (host OS)
FROM python:3.7

# set the dependencies file to the working directory
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .
COPY server.py .

# install dependencies
CMD ["echo", "ls"]
RUN pip install -r requirements.txt

# command to run on container start
ENTRYPOINT [ "python", "server.py" ]
