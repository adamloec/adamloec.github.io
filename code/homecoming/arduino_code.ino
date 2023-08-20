// Wire slave receiver for relay control

#include <Wire.h>

#define forward 10
#define backward 11

void setup() {
  Wire.begin(0x5); // join i2c bus with address #5
  // Wire.begin(0x6); // join i2c bus with address #6 (2nd arduino)
  // Wire.begin(0x7); // join i2c bus with address #7 (3rd arduino)
  // Wire.begin(0xN); // join i2c bus with address #8 (Nth arduino)
  Wire.onReceive(receiveEvent); // register event
  pinMode(forward, OUTPUT);
  pinMode(backward, OUTPUT);
}

void loop() {
  delay(100);
}

// function that executes whenever data is received from master
void receiveEvent(int howMany) {
  while (Wire.available()) { // if wire connection is available
    char c = Wire.read(); // receive byte as a character
    
    if (c == 1 && c != 2){
      digitalWrite(forward, c); // sets the forward relay to close when input is 1
    }
    else if (c == 2 && c != 1){
      digitalWrite(backward, c); // sets the backward relay to close when input is 2
    }
    else{
      digitalWrite(forward, c); // resets both relays to remain open when input is 0
      digitalWrite(backward, c);
    }
  }
}
