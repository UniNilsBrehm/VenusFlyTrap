import serial
import time
import tkinter as tk
import numpy as np
from scipy.io.wavfile import write as wavwrite
from pygame import mixer


# Function for left arrow key press
def left_arrow_pressed(event):
    cc_button_pressed()


# Function for right arrow key press
def right_arrow_pressed(event):
    ccw_button_pressed()


def send_trigger_via_key(event):
    # Send the trigger signal
    print("Trigger button pressed")
    ser.write(b'T')
    # sound.play()


def send_trigger():
    # Send the trigger signal
    print("Trigger button pressed")
    ser.write(b'T')
    # sound.play()


# Function for CC button
def cc_button_pressed():
    # Move ClockWise for one step
    print("CC button pressed")
    ser.write(b'R')


# Function for CCW button
def ccw_button_pressed():
    # Move CounterClockWise for one step
    print("CCW button pressed")
    ser.write(b'L')


# SOUND
# Sine Wave Settings
frequency = 50
duration = 2
amplitude = 4

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

# CONNECT TO ARDUINO
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino's serial port
time.sleep(2)  # Wait for the Arduino to initialize
print('Arduino connected!')

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

# Create Buttons
btn_height = 2
btn_width = 5
# Create a larger button and place it at the center
play_button = tk.Button(root, text='Trigger', command=send_trigger, height=btn_height, width=btn_width)
play_button.place(relx=0.3, rely=0.3, anchor=tk.CENTER)

sound_button = tk.Button(root, text='Sound', command=sound.play, height=btn_height, width=btn_width)
sound_button.place(relx=0.7, rely=0.3, anchor=tk.CENTER)

# Create CC button
cc_button = tk.Button(root, text='CC', command=cc_button_pressed, height=btn_height, width=btn_width)
cc_button.place(relx=0.3, rely=0.7, anchor=tk.CENTER)

# Create CCW button
ccw_button = tk.Button(root, text='CCW', command=ccw_button_pressed, height=btn_height, width=btn_width)
ccw_button.place(relx=0.7, rely=0.7, anchor=tk.CENTER)

# Bind left arrow key to CC button
root.bind('<Left>', left_arrow_pressed)

# Bind right arrow key to CCW button
root.bind('<Right>', right_arrow_pressed)

# Bind 'T' to send trigger
root.bind('t', send_trigger_via_key)

root.mainloop()
ser.close()  # Close the serial connection
