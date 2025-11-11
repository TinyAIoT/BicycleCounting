/*
  RadioLib LoRaWAN Starter Example

  ! Please refer to the included notes to get started !

  This example joins a LoRaWAN network and will send
  uplink packets. Before you start, you will have to
  register your device at https://www.thethingsnetwork.org/
  After your device is registered, you can run this example.
  The device will join the network and start uploading data.

  Running this examples REQUIRES you to check "Resets DevNonces"
  on your LoRaWAN dashboard. Refer to the network's 
  documentation on how to do this.

  For default module settings, see the wiki page
  https://github.com/jgromes/RadioLib/wiki/Default-configuration

  For full API reference, see the GitHub Pages
  https://jgromes.github.io/RadioLib/

  For LoRaWAN details, see the wiki page
  https://github.com/jgromes/RadioLib/wiki/LoRaWAN

*/

#include "config.h"
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Global variables
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("*** Device Connected! ***");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("*** Device Disconnected! ***");
      // Restart advertising when device disconnects
      pServer->startAdvertising();
    }
};

void sendLoRaUplink(String message) {
  Serial.println(F("Preparing LoRa uplink"));

  // Convert String to byte array
  uint8_t payload[message.length()];
  message.getBytes(payload, message.length() + 1);

  // Send it
  int16_t state = node.sendReceive(payload, sizeof(payload));

  debug(state < RADIOLIB_ERR_NONE, F("Error in sendReceive"), state, false);

  if (state > 0) {
    Serial.println(F("Received a downlink"));
  } else {
    Serial.println(F("No downlink received"));
  }

  Serial.println();
}

// Characteristic callback class for handling write operations
class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      String rxValue = pCharacteristic->getValue();

      if (rxValue.length() > 0) {
        Serial.println("*********");
        Serial.print("Received Value: ");
        Serial.println(rxValue);
        Serial.println("*********");

        // Send over LoRa
        sendLoRaUplink(rxValue);

        // Optional: respond to BLE client
        String response = "LoRa sent: " + rxValue;
        pCharacteristic->setValue(response.c_str());
        pCharacteristic->notify();
      }
    }
};


void setup() {
  Serial.begin(115200);

  Serial.println(F("\nSetin up LoRaWan connection ... "));

  Serial.println(F("Initialise the radio"));
  int16_t state = radio.begin();
  debug(state != RADIOLIB_ERR_NONE, F("Initialise radio failed"), state, true);

  // SX1262 rf switch order: setRfSwitchPins(rxEn, txEn);
  radio.setRfSwitchPins(38, RADIOLIB_NC);
  
  // Setup the OTAA session information
  state = node.beginOTAA(joinEUI, devEUI, nwkKey, appKey);
  // node.beginOTAA(joinEUI, devEUI, nwkKey, appKey);
  debug(state != RADIOLIB_ERR_NONE, F("Initialise node failed"), state, true);

  Serial.println(F("Join ('login') the LoRaWAN Network"));
  state = node.activateOTAA();
  debug(state != RADIOLIB_LORAWAN_NEW_SESSION, F("Join failed"), state, true);

  // while(1)
  // {
  //   state = node.activateOTAA(LORAWAN_UPLINK_DATA_RATE);
  //   if(state == RADIOLIB_LORAWAN_NEW_SESSION) break;
  //   debug(state!= RADIOLIB_LORAWAN_NEW_SESSION, F("Join failed"), state, true);
  //   delay(15000);
  // }

  // // Disable the ADR algorithm (on by default which is preferable)
  // node.setADR(false);

  // // Set a fixed datarate
  // node.setDatarate(LORAWAN_UPLINK_DATA_RATE);

  // // Manages uplink intervals to the TTN Fair Use Policy
  // node.setDutyCycle(false);

  Serial.println(F("LoRaWan is Ready!\n"));

  Serial.println("\n Starting BLE Server now.");

  // Create the BLE Device
  BLEDevice::init("ESP32_BLE_Server");

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create BLE Characteristic
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE  |
                      BLECharacteristic::PROPERTY_NOTIFY |
                      BLECharacteristic::PROPERTY_INDICATE
                    );

  pCharacteristic->setCallbacks(new MyCallbacks());

  // Create a BLE Descriptor
  pCharacteristic->addDescriptor(new BLE2902());

  // Set initial value
  pCharacteristic->setValue("Hello from ESP32 Server!\n");

  // Start the service
  pService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  
  Serial.println("BLE Server ready! Waiting for connections...");
  Serial.println("Service UUID: " + String(SERVICE_UUID));
  Serial.println("Characteristic UUID: " + String(CHARACTERISTIC_UUID));
}

void loop() {
  // Handle connection state changes
  if (!deviceConnected && oldDeviceConnected) {
    delay(500); // give the bluetooth stack the chance to get things ready
    oldDeviceConnected = deviceConnected;
  }
  
  if (deviceConnected && !oldDeviceConnected) {
    // do stuff here on connecting
    oldDeviceConnected = deviceConnected;
  }

  // Send periodic messages to connected client
  if (deviceConnected) {
    static unsigned long lastTime = 0;
    unsigned long currentTime = millis();
    
    // Send a message every 5 seconds
    if (currentTime - lastTime > 12000) {
      String message = "pong";
      pCharacteristic->setValue(message.c_str());
      pCharacteristic->notify();
      //Serial.println("Sent message: " + message);
      lastTime = currentTime;
    }
  }
  
  delay(1000);
} 