#include <CheapStepper.h>

CheapStepper stepper(8, 9, 10, 11);

boolean moveClockwise = false;
boolean moveCounterClockwise = false;
int number_of_steps = 8;
int mini_steps = 5;  // 10 steps are around 1 degree
int trigger_pin = 7;
boolean triggerReceived = false;  // Variable to check if trigger is received
boolean triggerL = false;
boolean triggerR = false;

void setup()
{
  Serial.begin(9600);  // Initialize serial communication
  pinMode(trigger_pin, OUTPUT);    // sets the digital pin as output
}

void loop()
{
  // Check for trigger signal
  if (Serial.available() > 0) {
    char trigger = Serial.read();
    if (trigger == 'T') {  // Assuming 'T' is your trigger signal
      triggerReceived = true;
    }
        if (trigger == 'L') {
      triggerL = true;
    }
        if (trigger == 'R') {
      triggerR = true;
    }
  }

  // If trigger is received, move stepper motor
  // full rotation == 4096 steps
  if (triggerReceived) {
    // Round 1
    digitalWrite(trigger_pin, HIGH); // sets the digital pin on
    for (int s = 0; s < number_of_steps; s++) {
      stepper.step(false);
    }

    delay(200);
    for (int s = 0; s < number_of_steps; s++) {
      stepper.step(true);
    }
    // moveClockwise = !moveClockwise;
    digitalWrite(trigger_pin, LOW); // sets the digital pin off
    
    delay(5000);
    
    // Round 2
    digitalWrite(trigger_pin, HIGH); // sets the digital pin on
    for (int s = 0; s < number_of_steps; s++) {
      stepper.step(false);
    }

    delay(200);
    for (int s = 0; s < number_of_steps; s++) {
      stepper.step(true);
    }
    // moveClockwise = !moveClockwise;
    digitalWrite(trigger_pin, LOW); // sets the digital pin off
    
    // Reset triggerReceived flag
    triggerReceived = false;
  }
  
  // Left (CCW) One Step
  if (triggerL) {
    for (int s = 0; s < mini_steps; s++) {
      stepper.stepCCW();
    }
    // Reset triggerL flag
    triggerL = false;
  }

  // Right (CW) One Step
  if (triggerR) {
    for (int s = 0; s < mini_steps; s++) {
      stepper.stepCW();
    }
    // Reset triggerL flag
    triggerR = false;
  }
  
}
