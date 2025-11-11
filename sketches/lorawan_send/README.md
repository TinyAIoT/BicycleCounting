# LoRaWAN + BLE Bridge (ESP32 + SX1262)

This project demonstrates how to bridge **BLE (Bluetooth Low Energy)** with **LoRaWAN uplinks** using an ESP32 and an SX1262 LoRa module.  
It allows BLE clients to send messages to the ESP32, which then forwards them via LoRaWAN to **The Things Network (TTN)** (or another LoRaWAN-compatible server).  

---

## Features
- **LoRaWAN OTAA Join** using `RadioLib`
- Uplink message transmission over LoRa
- BLE GATT Server with:
  - Custom **Service UUID**
  - Custom **Characteristic UUID**
  - Notifications to connected clients
- BLE → LoRa message forwarding
- Downlink reception support

---

## Repository Structure

```text
├── config.h       # Radio & LoRaWAN configuration (keys, EUI, region, etc.)  
├── main.ino       # Main sketch (LoRa + BLE logic)
```
---

## Requirements
- ESP32 board (e.g., XIAO ESP32S3, ESP32 DevKit, etc.)
- SX1262 LoRa module (SPI connection)
- [RadioLib](https://github.com/jgromes/RadioLib) library (we used v7.3.0)
- Arduino IDE or PlatformIO

---

## Hardware Connections (default example)

The radio module pins are configured as:
```cpp
SX1262 radio = new Module(41, 39, 42, 40); 
// NSS: 41, DIO1: 39, NRST: 42, BUSY: 40

// If you’re using a RadioBoard, you can enable:

#define RADIO_BOARD_AUTO
#include <RadioBoards.h>
Radio radio = new RadioModule();
```

⸻

### LoRaWAN Setup 
1.	Register your device on The Things Network Console.
	- Create a new end device
	- Select OTAA mode
	- Get your DevEUI, AppKey, and NwkKey
2.	Replace placeholders in config.h:

```cpp
#define RADIOLIB_LORAWAN_DEV_EUI   0x1234567890ABCDEF
#define RADIOLIB_LORAWAN_APP_KEY   0x00, 0x11, 0x22, ... , 0xFF
#define RADIOLIB_LORAWAN_NWK_KEY   0x00, 0x11, 0x22, ... , 0xFF
```

3.	Select your LoRaWAN region:

```cpp
const LoRaWANBand_t Region = EU868;  
// Options: EU868, US915, AU915, AS923, IN865, KR920, CN500
```

⸻

### BLE Setup
•	Service UUID: ``4fafc201-1fb5-459e-8fcc-c5c9c331914b``
•	Characteristic UUID: ``beb5483e-36e1-4688-b7f5-ea07361b26a8``

BLE client apps (like nRF Connect, LightBlue, or a custom app) can connect, write data, and receive notifications.

⸻

## How It Works
1.	ESP32 initializes the LoRa radio and joins the LoRaWAN network (OTAA).
2.	A BLE client connects to ESP32 and writes a message.
3.	ESP32 forwards the message over LoRaWAN uplink.
4.	Any received downlink is logged over Serial.
5.	ESP32 sends periodic BLE notifications (pong) every 12 seconds when connected.
