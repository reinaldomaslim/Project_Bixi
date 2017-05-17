const int sensor1 = A5;
const int sensor2 = 41;
const int power1 = A8;
const int power2 = 33;
bool value1, value2;

void setup() {
  // put your setup code here, to run once:
  pinMode(sensor1, INPUT);
  pinMode(sensor2, INPUT);
  pinMode(power1, OUTPUT);
  pinMode(power2, OUTPUT);
  digitalWrite(power1, HIGH);
  digitalWrite(power2, HIGH);
  
  Serial.begin(9600);
}

void loop() {
  value1 = digitalRead(sensor1);
  value2 = digitalRead(sensor2);
  Serial.print("Sensor1 - ");
  Serial.print(value1);
  Serial.print("\tSensor2 - ");
  Serial.print(value2);
  Serial.println();
  delay(100);
}
