# Use the official Python image as the base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y libsndfile1

RUN apt-get update && \
    apt-get install -y libportaudio2

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg

# Install espeak and its dependencies
RUN apt-get update && \
    apt-get install -y libespeak1


# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install protobuf==3.20

RUN pip install -r requirements.txt

# Copy your application code to the container
COPY app16.py .

# Copy any additional files or directories your application requires
COPY AllAudio /app/AllAudio

COPY fyplabels.csv /app/fyplabels.csv

# Copy any other dependencies or configuration files

# Install Python dependencies
RUN pip install Flask librosa numpy tensorflow pandas scikit-learn pydub soundfile SpeechRecognition pyttsx3


# Start your Flask application
CMD ["python", "app16.py", "--host", "0.0.0.0", "--port", "5000"]
