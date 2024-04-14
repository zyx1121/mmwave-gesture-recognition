# mmWave Radar AI Gesture Recognition

###### Gesture Recognition Using mmWave Sensor - TI AWR1642

This project provides Setup, Record, Train, and Predict functionalities to identify specific gestures using the AWR1642 mmWave radar data. Supported gestures include:

- Swipe Up
- Swipe Down
- Swipe Left
- Swipe Right
- Clockwise Rotation
- Counterclockwise Rotation

> The `Record` feature allows you to capture gestures for training.

## Prerequisites

Flash the official demo firmware onto the AWR1642 development board before starting.

- [Initial Setup for AWR1642BOOST](https://gist.github.com/zyx1121/0756055fa9138aec81617501e2e5f263)

## Getting Started

- Clone the repository and cd to the project directory
  ```sh
  git clone https://github.com/zyx1121/mmwave.gesture.recognition && cd mmwave.gesture.recognition
  ```

- Install dependencies
  ```sh
  pip install -r requirements.txt
  ```

- Launch the console to begin
  ```sh
  python console.py
  ```

## Project Structure

- `mmwave.gesture.recognition/`
  - `models/`
    - `Conv2D.keras`
    - `LSTM.keras`
  - `records/`
    - `[label]_[%m%d%H%M%S].npy`
  - `console.py`
  - `mmwave.py`
  - `profile.cfg`

## Command Functions

- `cfg` : Transmits settings from profile.cfg to the device.

- `record` `[gesture]` `[times]` : Records the gesture [gesture] data [times] times and saves it to records/[gesture]_[date].npy.

- `train` `[model]` : Trains the model, with a choice of either Conv2D or LSTM.

- `predict` `[model]` : Captures real-time radar data and predicts gestures using the selected model (Conv2D or LSTM).

- `exit` : Exits the console.

## License

This project is licensed under the MIT License, which permits commercial use, modification, distribution, and private use.
