# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install Git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and other necessary files
COPY ./pyproject.toml /app/
COPY ./README.md /app/
COPY ./LICENSE /app/

# Copy the current directory contents into the container at /app
COPY . /app/

# Copy the .git directory for versioning
COPY .git /app/.git/

# Install Flit and other build dependencies
RUN pip install flit

# Set environment variable to allow flit to install as root
ENV FLIT_ROOT_INSTALL=1

# Install project dependencies
RUN flit install --deps production

# Make port 80 available to the world outside this container
EXPOSE 80

# Run entrypoint.py when the container launches
CMD ["python", "crescendo/entrypoint.py"]