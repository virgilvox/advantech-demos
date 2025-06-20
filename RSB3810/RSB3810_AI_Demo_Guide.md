# RSB‑3810 Ubuntu Quick‑Start & Edge‑AI Demo

<img src="./images/main.png" width="700"/>

---

## Introduction

The Advantech RSB‑3810 is a compact yet powerful edge AI platform built around the MediaTek Genio 1200 SoC, offering an efficient mix of high-performance CPU cores, an integrated GPU, and a 4.8 TOPS NPU. It's designed to bridge the gap between development flexibility and production readiness for smart embedded systems.

In industries increasingly driven by real-time AI—whether in automation, robotics, retail, or smart cities—developers face a common bottleneck: how to prototype quickly on production-grade hardware without the friction of complex BSPs or opaque toolchains. The RSB‑3810 solves this with robust upstream support, a full Ubuntu 22.04 Linux environment, and tooling like genio-tools that make flashing and debugging simple.

Whether you're building vision-based applications, deploying lightweight ML inference, or integrating high-speed industrial I/O, the RSB‑3810 delivers a rare balance of performance, connectivity, and openness—making it a compelling choice for modern AIoT workflows.

## Board Specs

- **MediaTek Genio 1200 (MT8395) SoC**
  - 4x Arm Cortex-A78 + 4x Cortex-A55
  - Mali-G57 GPU
  - Integrated 4.8 TOPS AI accelerator (APU 3.0)
- **Memory & Storage**
  - Up to 8GB LPDDR4X
  - 64GB eMMC onboard
  - microSD slot available
- **Video I/O**
  - HDMI 2.0 (4K@60Hz output)
  - MIPI-CSI for camera input
  - MIPI-DSI for LCD panel output
- **Connectivity**
  - Gigabit Ethernet
  - Optional Wi-Fi/BT module via M.2 Key E
  - 4G/5G support via M.2 Key B
- **Expansion Interfaces**
  - 1x USB 3.0, 1x USB 2.0, 1x USB OTG
  - UART, I2C, SPI, CAN, GPIO
  - M.2 Key M for storage expansion
  - PCIe Gen3 (1-lane)
- **Power & Form Factor**
  - Wide voltage input: 12V DC
  - Industrial-grade SBC form factor (Pico-ITX 100mm x 72mm)
- **Software Support**
  - Ubuntu 22.04 LTS
  - Android 11
  - Genio SDK / Genio Tools
  - U-Boot and Linux Kernel upstream support

## Overview

In this guide we'll walk through flashing Ubuntu 22.04 to the Advantech [RSB‑3810](https://ess-wiki.advantech.com.tw/view/RSB-3810), installing the MediaTek genio‑tools SDK, and running a real‑time TensorFlow Lite object‑detection demo. By the end you'll have a fan‑less edge computer recognising objects from a USB webcam—ready for industrial IoT or smart‑vision prototypes.

## Prerequisites

- [RSB‑3810 board](https://ess-wiki.advantech.com.tw/view/RSB-3810)
- 12V DC power adapter
- USB‑A to USB‑A cable (for flashing)
- USB-to-RS232 serial adapter + included DB9 cable (female-female)
- Ubuntu 22.04 system (native or VM with USB passthrough enabled)
- [Genio tools installation guide](https://mediatek.gitlab.io/aiot/doc/aiot-dev-guide/master/sw/yocto/get-started/env-setup/flash-env-linux.html)

---

## Flashing the Board (USB Fastboot)

### 1. Install Genio Tools

Follow the [official genio-tools instructions](https://mediatek.gitlab.io/aiot/doc/aiot-dev-guide/master/sw/yocto/get-started/env-setup/flash-env-linux.html) and ensure the USB rules and permissions are correctly set.

```bash
sudo apt update
pipx install genio-tools
# or: pip3 install genio-tools --break-system-packages
```
<img src="./images/sudo-apt-update-ubuntu-for-genio.png" width="600"/>

Add udev rules:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="201c", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0e8d", ATTR{idProduct}=="0003", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="gpio", MODE="0660", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/72-aiot.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -a -G plugdev $USER
```
<img src="./images/set-udev-rules.png" width="600"/>

Log out and back in.

### 2. Get the OS Image

Download the latest image from the [RSB‑3810 Wiki](https://advantecho365-my.sharepoint.com/:u:/g/personal/jems_cheng_advantech_com/EZkFOqGDCL9MgE_PAdHvhIIBgjPQU9bqtHtFUM1i_WF-vA?e=IQGq1ds) — Ubuntu 22.04 eMMC image.

```bash
tar -xvzf Ubuntu_Recovery_Image.tgz -C ~/ 
cd ~/rsb3810_ubuntu
```
<img src="./images/unzip-recovery-image.png" width="600"/>

### 3. Flash Procedure

1. Connect the USB-A ↔ USB-A cable to the board and your Ubuntu machine
    <img src="./images/usb-a-plugged-to-flash-port.png" width="600"/>
2. Power on the board (12V DC)
    
    <img src="./images/plug-in-power.png" width="600"/>
3. In terminal, run:

```bash
genio-flash
```
<img src="./images/run-genio-flash.png" width="600"/>

4. On the board, **hold the Download button**, **press Reset**, then **release Download** when < wait for any device > appears onscreen. 
    <img src="./images/download-buton-position.png" width="600"/>
    <img src="./images/reset-button-position.png" width="600"/>
5. When you see `Waiting for any device`, wait ~8–10s for the flashing to begin

Full flash takes several minutes. Wait for completion confirmation.
<img src="./images/genio-flash-complete.png" width="600"/>

---

## Debug Console (UART Serial Access)

### Connection Setup

Use the included DB9 cable with the USB-to-RS232 adapter. The correct DB9 connector is the one with **3 physical pins visible**.

1. Connect DB9 to the RS232 port on the board (labeled COM1)
    <img src="./images/plug-serial-to-board.png" width="600"/>
2. Plug adapter into your host machine
3. Use `screen` or `minicom` at `921600` baud:

```bash
screen /dev/ttyUSB0 921600
```

> You may need to hit `Enter` a few times if output is garbled at first.

### Tip for VM Users

Ensure USB passthrough is enabled for the RS232 adapter in VirtualBox or VMware. Add a device filter manually if needed.

For USB flashing in a VM: make sure the Mediatek device is passed through to the VM before running `genio-flash`. It may only appear briefly when booting to flash mode.

---

## Run the TensorFlow Lite AI Demo

### 1. Network Access

If your RSB‑3810 doesn't include Wi‑Fi, plug in Ethernet. Internet is required to install dependencies.
<img src="./images/ethernet-plugged-in.png" width="600"/>

### 2. Log In

Use UART console or attach monitor + keyboard. Default login credentials:

```bash
Username: ubuntu
Password: ubuntu
```

> If prompted to change the password but it fails with `Authentication token manipulation error`, log in without sudo and use:

```bash
passwd
```

Then re-login to continue.

### 3. Set Up Demo Environment

Install video and capture dependencies to support most USB webcams:

```bash
sudo apt install v4l-utils libv4l-dev libjpeg-dev libgstreamer1.0-dev                  gstreamer1.0-plugins-base gstreamer1.0-plugins-good                  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly                  gstreamer1.0-libav ffmpeg
```

Then install Python dependencies:

```bash
sudo apt update
sudo apt install python3-pip python3-opencv -y
pip3 install numpy==1.24.4 tflite-runtime
```

#### Alternative: Automated Install

As a convenience, an `install.sh` script is provided to automate the setup steps. After logging into the board, you can run it directly:

```bash
./install.sh
```
This will install all necessary packages and download the AI model files.

### 4. Get Demo Script

Use a compatible MobileNet SSD model directly from TensorFlow:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
mv detect.tflite model.tflite
mv labelmap.txt labels.txt
```
<img src="./images/download-unzip-model.png" width="600"/>

`labelmap.txt` already starts with "???" (the background class), so no edits are required.

Then fetch or create the demo script:

```bash
nano demo.py
```

```python
import cv2
import numpy as np
import sys
import tflite_runtime.interpreter as tflite

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f if line.strip()]

if labels and labels[0].startswith('???'):
    labels[0] = 'background'

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

video_device = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'
cap = cv2.VideoCapture(video_device)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:
            cid = int(classes[i]) + 1
            label = labels[cid] if cid < len(labels) else f"Unknown ({cid})"
            print(f"Detected ID {cid}: {label} at {scores[i]*100:.1f}% confidence")

    if hasattr(cv2, 'imshow'):
        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, top, right, bottom) = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]),
                                              int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
                cid = int(classes[i]) + 1
                label = labels[cid] if cid < len(labels) else f"Unknown ({cid})"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {scores[i]*100:.1f}%", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('RSB‑3810 Live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

Then run the demo:

```bash
python3 demo.py
# or, specify camera device:
python3 demo.py /dev/video5
```
<img src="./images/gui-object-detection.png" width="600"/>
<img src="./images/gui-object-detection2.png" width="600"/>


## Final Thoughts

The RSB‑3810 offers an accessible and efficient path into edge AI development. From its robust Genio 1200 SoC with integrated NPU, to its rich I/O and flexible expansion options, the board delivers both power and versatility in a small footprint. Whether you’re experimenting with computer vision, deploying smart sensors, or building a kiosk or robotics solution, this board enables you to develop and test with minimal friction. With proper Linux support, open tooling, and solid documentation, the RSB‑3810 is ready for both prototyping and real-world deployment.
---

## Optional: Interactive Development with Jupyter

For a more interactive development experience, you can install and run JupyterLab on the RSB-3810. This allows you to edit and run Python code from a web browser on another computer on the same network.

### 1. Install JupyterLab

From the terminal on your RSB-3810, run the following command:

```bash
pip3 install jupyterlab
```

### 2. Run JupyterLab

To start the Jupyter server, run:

```bash
jupyter lab --ip=0.0.0.0
```

This will start the server and make it accessible from other devices on your local network. It will output a URL containing a security token. Copy the full URL and paste it into a web browser on your development machine to access the Jupyter interface.

Example URL:
`http://<rsb-3810-ip-address>:8888/lab?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxx`

You can now create new notebooks (`.ipynb` files) and run the demo code interactively, cell by cell, which is great for debugging and experimentation.

---

## Resources

- [RSB‑3810 ESS Wiki](https://ess-wiki.advantech.com.tw/view/AIM-Linux/RSB-3810)
- [Demo Project GitHub]()