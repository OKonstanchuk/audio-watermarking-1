import librosa
import pickle
import numpy as np
import os
# audio_path = librosa.ex("/media/tongch/C8B41AFCB41AED26/MusicSamples/cls_Canon_In_D.mp3")
dir_path = "/media/tongch/C8B41AFCB41AED26/MusicSamples/"
# audio_path = "/media/tongch/C8B41AFCB41AED26/MusicSamples/cls_Canon_In_D.mp3"
samplerate = 44100
directory = os.fsencode(dir_path)

# generate audio examples
# with open("audio_examples.pkl",'wb') as f:
#     for file in os.listdir(directory):
#          filename = os.fsdecode(file)
#          audio_path = dir_path+filename
#          print(dir_path+filename)
#          # read current audio
#          soundwave,samplerate = librosa.load(audio_path,samplerate)
#          total_sections = int(len(soundwave)/samplerate)
#          # print(len(soundwave),total_sections)
#          for i in range(total_sections):
#              start, end = i*samplerate,(i+1)*samplerate
#              pieces = soundwave[start:end]
#              if len(pieces) == samplerate:
#                  pickle.dump(pieces,f)
#              print(len(pieces))

# load examples
audios = []
with open("audio_examples.pkl",'rb') as f:
    while True:
        try:
            x = pickle.load(f)
            audios.append(x)
        except EOFError:
            break

print(len(audios))
# watermarkings = []
