import winsound
import time

freq = 40
duration = 2000  # in ms
print('Enter "exit" to exit program')
print(f'Beep Duration is set to: {duration/1000:.2f}s ')
print(f'Beep Frequency is set to: {freq:.2f} Hz (range 37-32767 Hz)')
print('')

msg = input('Press Enter to Beep: ')

while msg != 'exit':
    print('.... BEEEEEPPPP ....')
    winsound.Beep(freq, duration)
    time.sleep(1)
    print('')
    msg = input('Press Enter to Beep: ')
    print('')

print('Exit Program')

