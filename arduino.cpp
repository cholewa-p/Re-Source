#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "siec_wifi";
const char* password = "123456789";
const char* serverName = "http://192.168.1.100:5000/data/update_power_generation_data";
const int DEVICE_ID = 1234;
StaticJsonDocument<1024> dataBuffer;

void setup() {
  Serial.begin(115200);

  dataBuffer.to<JsonArray>();

  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  delay(60000);


  float power_kw = random(50, 300) / 100.0;

  JsonArray array = dataBuffer.as<JsonArray>();
  JsonObject newPoint = array.createNestedObject();
  newPoint["timestamp"] = String(millis()); 
  newPoint["device_id"] = DEVICE_ID;
  newPoint["power"] = power_kw;

  Serial.print("New reading added to buffer. Buffer size: ");
  Serial.println(array.size());


  if (WiFi.status() == WL_CONNECTED) {

    if (array.size() == 0) {
      Serial.println("Buffer is empty, nothing to send.");
      return;
    }

    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String jsonPayload;
    serializeJson(dataBuffer, jsonPayload);

    Serial.print("Attempting to send payload: ");
    Serial.println(jsonPayload);

    int httpResponseCode = http.POST(jsonPayload);

    if (httpResponseCode == 200) { 
      String response = http.getString();
      Serial.println("Payload sent successfully!");
      Serial.print("Response: ");
      Serial.println(response);
      
      dataBuffer.clear();
      dataBuffer.to<JsonArray>(); 
      Serial.println("Buffer cleared.");

    } else {
      Serial.print("Error on sending POST: ");
      Serial.println(httpResponseCode);
      Serial.println("Data will be sent on next attempt.");
    }

    http.end();
  } else {
    Serial.println("WiFi Disconnected. Data stored in buffer.");
  }
}