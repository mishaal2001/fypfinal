import ctypes

try:
    sndfile_lib = ctypes.CDLL('libsndfile.so')
    print("libsndfile library loaded successfully.")
except Exception as e:
    print("Error loading libsndfile library:", e)
