# mmWave Radar AI Gesture Recognition

###### Gesture Recognition Using mmWave Sensor - TI AWR1642

![gif](https://github.com/zyx1121/mmwave-gesture-recognition/assets/98001197/b271cdad-25c1-46b4-af98-e218882e354b)

This project provides Setup, Record, Train, and Predict functionalities to identify specific gestures using the AWR1642 mmWave radar data. Supported gestures include:

- Swipe Up
- Swipe Down
- Swipe Left
- Swipe Right
- Clockwise Rotation
- Counterclockwise Rotation

> [!NOTE]
> The `Record` feature allows you to capture gestures for training.

## Prerequisites

- Flash the official demo firmware onto the AWR1642 development board before starting.
    [Initial Setup for AWR1642BOOST](https://gist.github.com/zyx1121/0756055fa9138aec81617501e2e5f263)

- Get `poetry` to manage the dependencies.
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

## Getting Started

- Clone the repository and cd to the project directory
    ```sh
    git clone https://github.com/zyx1121/mmwave-gesture-recognition && cd mmwave-gesture-recognition
    ```

- Install dependencies
    ```sh
    poetry install
    ```

- Launch the console to start
    ```sh
    poetry run mmwave-gesture-recognition
    ```

- Setup the AWR1642 development board
    ```sh
    CLI > cfg
    ```

- Train the model
    ```sh
    CLI > train LSTM
    ```

- Plot the radar data in real-time and press `q` to exit
    ```sh
    CLI > plot
    ```

- Start predicting gestures and press `ctrl+c` to exit
    ```sh
    CLI > predict
    ```

## Project Structure

- `mmwave-gesture-recognition/`
    - `configs/` - (Configuration files)
        - `profile.cfg`
    - `models/` - (Trained models)
        - `Conv2D.keras`
        - `LSTM.keras`
    - `records/` - (Gesture data records)
        - `[label]_[%m%d%H%M%S].npy`
        - `...`
    - `src/` - (Source code)
        - `console/` - (Console interface)
            - `cli.py` - (Command-line interface)
        - `mmwave/` - (mmWave radar data processing)
            - `radar.py` - (Radar data processing)
            - `uart.py` - (UART communication)
        - `utils/` - (Utility functions)
            - `logger.py` - (Logging configuration)
        - `main.py` - (Main entry point)
    - `pyproject.toml` - (Poetry configuration)
    - `README.md` - (Project documentation)

## Command Functions

- `cfg` : Transmits settings from profile.cfg to the device.

- `plot` : Plots the radar data in real-time.

- `record` `[gesture]` `[times]` : Records the gesture [gesture] data [times] times and saves it to records/[gesture]_[date].npy.

- `train` `[model]` : Trains the model, with a choice of either Conv2D or LSTM.

- `predict` `[model]` : Captures real-time radar data and predicts gestures using the selected model (Conv2D or LSTM).

- `clear` : Clears the console screen.

- `exit` : Exits the console.

## License

This project is licensed under the MIT License, which permits commercial use, modification, distribution, and private use.
