

import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import ctypes
ctypes.CDLL('libsndfile.so')
import os
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))


# Define the input shape of the audio data
input_shape = (431, 128, 1)  # because n_mels = 128
max_time_steps = 431  # originally 300
num_features = 128
train_time_series_data = 38400

#from google.colab import drive
#drive.mount('/content/drive')

# Load the audio files and extract features
audio_dir = "AllAudio"
audio_files = os.listdir(audio_dir)
audio_features = []
for file in audio_files:
    audio_path = os.path.join(audio_dir, file)
    audio_data, sr = librosa.load(audio_path, sr=22050, mono=True, duration=10)
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=num_features)
    log_spectrogram = librosa.power_to_db(spectrogram)
    log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)
    audio_features.append(log_spectrogram)
audio_features = np.array(audio_features)



# Reshape the input data to match the expected input shape
input_data = np.transpose(np.expand_dims(log_spectrogram, axis=0), (0, 2, 1, 3))

# Load the stuttering labels
labels = pd.read_csv("fyplabels.csv")

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(audio_features, labels, test_size=0.3, random_state=0)

# Transpose the data to match the RNN input shape
train_data = np.transpose(train_data, (0, 2, 1, 3))
test_data = np.transpose(test_data, (0, 2, 1, 3))

# Define the CNN architecture
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())

# Define the RNN architecture
rnn_input = Input(shape=(max_time_steps, num_features))
rnn_model = LSTM(units=64, return_sequences=False)(rnn_input)

# Combine the CNN and RNN models
combined_model = tf.keras.layers.concatenate([cnn_model.output, rnn_model])
output_layer = Dense(1, activation='sigmoid')(combined_model)
model = Model(inputs=[cnn_model.input, rnn_input], outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_data, np.random.rand(train_data.shape[0], max_time_steps, num_features)], train_labels, epochs=2,
          batch_size=32, validation_split=0.3)

#pickle.dump(model, open("model.pkl", "wb"))

from flask import Flask, request, jsonify
import sounddevice as sd
import soundfile as sf
import numpy as np
import difflib
import re
import speech_recognition as sr
import pyttsx3
import pydub
import subprocess
import traceback
from pydub import AudioSegment
from pydub.playback import play
from collections import defaultdict
from flask import render_template

app = Flask(__name__)

# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()







app = Flask(__name__)





def determine_stuttering_level(audio_file_path):
    # Load the audio file and extract features
    audio_data, sr = librosa.load(audio_file_path, sr=22050, mono=True, duration=10)
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=num_features)
    log_spectrogram = librosa.power_to_db(spectrogram)
    log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)

    # Reshape the input data to match the expected input shape
    input_data = np.transpose(np.expand_dims(log_spectrogram, axis=0), (0, 2, 1, 3))
    input_data = np.pad(input_data, ((0, 0), (0, max_time_steps - input_data.shape[1]), (0, 0), (0, 0)), mode='constant')

    # Make prediction using the model
    prediction = model.predict([input_data, np.random.rand(1, max_time_steps, num_features)])

    # Determine the stuttering level
    if prediction[0][0] < 0.33:
        stuttering_level = "Low stuttering"
    elif prediction[0][0] < 0.67:
        stuttering_level = "Medium stuttering"
    else:
        stuttering_level = "High stuttering"

    return stuttering_level






# Define a route for recording audio
@app.route('/record-audio', methods=['POST'])
def record_audio():
    global current_level
    try:
        # Set the audio parameters
        duration = 15  # Recording duration in seconds
        sample_rate = 44100  # Sample rate in Hz

        # Start recording
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        # Save the recorded audio to a WAV file
        file_path = 'C:\\21st-july-2023\\fypfinall\\recorded_audio.wav'
        sf.write(file_path, audio_data, sample_rate, 'PCM_16')

        # Perform speech recognition on the recorded audio
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)

        # Recognize the speech from the audio
        recorded_text = recognizer.recognize_google(audio_data)

        # Compare the recorded_text with the example_sentence and create HTML with red underline and pronunciation suggestions
        example_sentence = example_sentences[current_level - 1]
        example_words = example_sentence.split()
        recorded_words = recorded_text.split()

        html_code = '<br><h1>Recorded Text</h1><p>'

        # Initialize mispronounced_words list
        mispronounced_words = []

        for example_word, recorded_word in zip(example_words, recorded_words):
            if example_word == recorded_word:
                html_code += recorded_word + ' '
            else:
                pronunciation_suggestion = get_pronunciation_suggestion(recorded_word)
                if pronunciation_suggestion:
                    mispronounced_words.append((recorded_word, pronunciation_suggestion[0]))
                    html_code += f'<span style="text-decoration: underline red;">{recorded_word}</span> '
                else:
                    html_code += recorded_word + ' '

        # Determine the stuttering level
        stuttering_level = determine_stuttering_level(file_path)

        if not mispronounced_words:
            # Move to the next level if there are no mispronounced words
            current_level += 1
            if current_level > levels:
                html_code += '<h2>Congratulations! You have completed all levels.</h2>'
            else:
                html_code += f'''
                            <h2>Congratulations! Level {current_level - 1} completed.</h2>
                            <h1>Level {current_level}</h1>
                            <h2>Please read the following sentence: '{example_sentences[current_level - 1]}'</h2>
                            <form action="/record-audio" method="POST">
                                <input type="submit" value="Start Recording">
                            </form>
                        '''
        else:
            # Display pronunciation suggestions and re-record option
            html_code += '<h2>Pronunciation Suggestions:</h2>'
            for mispronounced_word, suggestion in mispronounced_words:
                html_code += f"<p>For '{mispronounced_word}', you may pronounce it like '{suggestion}'</p>"
            html_code += '''
                        <h2>Re-record Audio:</h2>
                        <form action="/record-audio" method="POST">
                            <input type="submit" value="Re-record">
                        </form>
                    '''

        # Prepare the response HTML page
        html_code = f'''
                              <!DOCTYPE html>
                              <html>
                              <head>
                                  <title>Recorded Audio and Text</title>
                              </head>
                              <body>
                              <h2>Stuttering Level: {stuttering_level}</h2>
                                  {html_code}
                              </body>
                              </html>
                              '''

        return html_code, 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400




if __name__ == '__main__':
    app.run(debug=True)
