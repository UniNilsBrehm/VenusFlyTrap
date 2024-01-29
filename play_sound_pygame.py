import tkinter as tk
import numpy as np
from scipy.io.wavfile import write as wavwrite
from pygame import mixer

# Sine Wave Settings
frequency = 400
duration = 1
amplitude = 1

# Generate the sine wave
t = np.arange(int(44100 * duration)) / 44100
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Convert the NumPy array to int16 format (required by simpleaudio)
wave_int16 = (wave * 32767).astype(np.int16)

# Save the wave to a temporary WAV file
temp_file = 'temp.wav'
wavwrite(temp_file, 44100, wave_int16)

# Create pygame sound mixer
mixer.init()
sound = mixer.Sound(temp_file)

# Create Tk Window
root = tk.Tk()

# Make the window larger
window_width = 400
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_position = int((screen_width - window_width) / 2)
y_position = int((screen_height - window_height) / 2)

root.geometry(f'{window_width}x{window_height}+{x_position}+{y_position}')

# Create a larger button and place it at the center
play_button = tk.Button(root, text='Play', command=sound.play, height=5, width=10)
play_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

root.mainloop()
