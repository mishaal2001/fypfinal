#import verify_libsndfile

import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import ctypes.util
from flask import Flask, render_template, request
import ctypes




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
import io
from flask import Flask, request, jsonify, render_template
import tempfile
import os



app = Flask(__name__)


# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

@app.route('/level')
def level():
    return render_template('level.html', level=current_level, example_sentence=example_sentence)

# Global variables for levels and example sentences
levels = 7
current_level = 1
example_sentences = [
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "the committee will meet next tuesday to discuss the new proposal",
    "the constitutional right to freedom of speech is protected by law",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
    "peter piper picked a peck of pickled peppers",
    "the mississippi river stretches from minnesota to louisiana"
]

# Specify the path to the ffmpeg executable
pydub.AudioSegment.ffmpeg = "C:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe"  # Update with the correct path
pydub.AudioSegment.converter = "C:\\sox\\sox.exe"  # Update with the correct path to sox




# Helper function to create HTML with red underline for unmatched words
def highlight_unmatched_words(example_sentence, recorded_text):
    example_words = example_sentence.split()
    recorded_words = recorded_text.split()
    html_code = '<br><h1>Recorded Text</h1><p>'

    for example_word, recorded_word in zip(example_words, recorded_words):
        if example_word == recorded_word:
            html_code += recorded_word + ' '
        else:
            html_code += f'<span style="text-decoration: underline red;">{recorded_word}</span> '

    html_code += '</p>'
    return html_code

from difflib import get_close_matches

# Helper function to find closest pronunciation match for a word
def get_pronunciation_suggestion(word):
    # Replace this with your pronunciation dictionary or API call for pronunciation suggestions
    # For now, we will use a dummy list with similar-sounding words
    pronunciation_dict = {
        "fox": ["foks", "fucks", "foksie", "foal", "fooox"],
        "jumps": ["jamps", "jumpsy", "jumpsie"],
        "dog": ["dawg", "doggie", "doggy", "d-d-d-o-g"],
        "she": ["shi", "shii", "shay"],
        "mississippi" : ["missing", "missippi", "missypi", "messysippi"],
        "the" : ["tha", "thu", "da", "they"],
        "quick" : ["kick", "quit", "quwick", "wick", "qick" "qui", "quuuick", "coil", "quill", "full"],
        "brown" : ["bown", "brow", "braun", "braawn", "rouwn", "frown"],
        "over" : ["ove", "owar", "ova", "overrr"],
        "lazy" : ["lady", "laze", "lizi", "lazii", "laasy", "lacy", "lucy"],
        "sells" : ["sails", "sales", "salls", "sellz","sell"],
        "seashells" : ["t-shirts","shesells", "sheshells", "seasells", "seashell"],
        "on" : ["one", "ooon", "onnn","oh", "ow"],
        "seashore" : ["sheshore", "seasore", "shesore", "shore", "sea", "she", "sure", "seaaashurrr"],
        "committee" : ["commit", "kitty" "cooomittee", "mittee", "commmmittee"],
        "will": ["well", "wail", "whale", "wing", "fill", "wwwill", "wheel","veil", "wall"],
        "meet": ["beat", "feet", "neat", "lead", "might", "myth", "met", "meek"],
        "next": ["nice", "nest", "egss", "naps", "nats", "nacks", "legs"],
        "tuesday": ["tweezday", "twosday", "teesday", "today", "wednesday", "thursday"],
        "to": ["do", "you", "top", "sew", "so", "new"],
        "discuss": ["curse", "diss", "cuss", "biscuit", "viscous", "ruckus", "daycuss"],
        "new" : ["sew", "few", "knee", "dew", "no", "ew"]

        # Add more words and their pronunciation suggestions as needed
    }
    return get_close_matches(word, pronunciation_dict.keys(), n=1)


# Import the necessary libraries
# ... (previous code remains the same)

# Define the Flask app
app = Flask(__name__)

# Global variables for levels and example sentences
# ... (previous code remains the same)

# Define the input shape and other audio-related variables
# ... (previous code remains the same)

# Load the stuttering model
# ... (previous code remains the same)

# Define a route for the home page

def determine_stuttering_level(recorded_audio_data):
    # Convert the recorded audio data to numpy array
    audio_data = np.frombuffer(recorded_audio_data, dtype=np.float32)

    # Resample the audio data to the desired sample rate
    target_sr = 22050  # Choose your desired sample rate
    audio_data_resampled = librosa.resample(audio_data, orig_sr=44100, target_sr=target_sr)

    # Extract features from the resampled audio data
    spectrogram = librosa.feature.melspectrogram(y=audio_data_resampled, sr=target_sr, n_mels=num_features)
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




@app.route('/')
def home():
    global current_level
    if current_level > levels:
        return "Congratulations! You have completed all levels."
    else:
        example_sentence = example_sentences[current_level - 1]

    html_code = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Level {current_level}</title>
    </head>
    <body>
        <h1>Level {current_level}</h1>
        <h2>Please read the following sentence: '{example_sentence}'</h2>
        <form action="/record-audio" method="POST">
            <input type="submit" value="Start Recording">
        </form>
    </body>
    </html>
    '''

    return html_code



# Define a route for recording audio
@app.route('/record-audio', methods=['POST'])
def record_audio():
    global current_level
    try:
        recorded_audio_data = request.files['audio'].read()
        # Recognize the speech from the audio
        recorded_text = recognizer.recognize_google(recorded_audio_data)

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

        # Determine the stuttering level using the recorded audio data
        stuttering_level = determine_stuttering_level(recorded_audio_data)

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



# Initialize the speech recognition engine
recognizer = sr.Recognizer()



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

