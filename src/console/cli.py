"""Command line interface module."""

import os
import cmd
import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv2D, Flatten
from keras.utils import to_categorical
from rich.progress import track

from ..mmwave.radar import MMWave
from ..utils.logger import logger, console

# Model configuration
MODEL_CONFIG = {
    "LSTM": {
        "name": "LSTM",
        "batch_size": 32,
        "epochs": 300,
        "validation_split": 0.2,
        "sequence_length": 20,
        "hidden_units": 128,
    },
    "Conv2D": {
        "name": "Conv2D",
        "batch_size": 32,
        "epochs": 300,
        "validation_split": 0.2,
        "sequence_length": 20,
        "filters": 32,
        "kernel_size": (3, 1),
    },
}

# Gesture labels definition
GESTURES = {
    "left": 0,  # Left swing
    "right": 1,  # Right swing
    "up": 2,  # Up swing
    "down": 3,  # Down swing
    "cw": 4,  # Clockwise
    "ccw": 5,  # Counterclockwise
}

# Generate reverse mapping from GESTURES
GESTURE_MAP = {num: label for label, num in GESTURES.items()}

# Constants related to gesture data collection
GESTURE_COLLECTION = {
    "INITIAL_FLAG": -1,  # Initial flag value
    "RESET_FLAG": 4,  # Reset flag value
    "END_FLAG": 0,  # End flag value
    "POSITION_THRESHOLD": 0.3,  # Position change threshold
}


class Console(cmd.Cmd):
    """Command line interface class.

    Provides a command line interface for interacting with the user.

    Attributes:
        prompt (str): Command prompt
        mmwave (MMWave): Radar controller instance
    """

    def __init__(self) -> None:
        """Initialize the command line interface."""
        super().__init__()
        self.prompt = "CLI > "

        # Set up matplotlib backend
        mpl.use("Qt5Agg")
        mpl.rcParams["toolbar"] = "None"

        self._init_radar()

    def _init_radar(self) -> None:
        """Initialize the radar device."""
        logger.info("Finding devices...")
        ports = MMWave.find_ports()

        if len(ports) < 2:
            logger.error("No devices found")
            exit(0)

        cli_port, data_port = ports[0], ports[1]
        self.mmwave = MMWave(cli_port, data_port)
        self.mmwave.connect()
        logger.info(f"Connected successfully CLI: {cli_port} DATA: {data_port}")

    def _validate_args(self, args: str, valid_args: list) -> None:
        """Validate command arguments.

        Args:
            args: Command arguments
            valid_args: List of valid arguments

        Raises:
            ValueError: Raised when arguments are invalid
        """
        if args not in valid_args:
            raise ValueError(f"Arguments must be one of {valid_args}")

    def do_cfg(self, args: str) -> None:
        """Configure the radar device.

        Args:
            args: Configuration file name, defaults to 'profile'
        """
        args = args or "profile"

        logger.info("Setting up device...")
        config_file = Path("configs") / f"{args}.cfg"
        if not self.mmwave.send_configure(config_file):
            logger.error("Configuration failed")
            return
        logger.info("[green]Configuration completed[/green]")

    def do_plot(self, args: str) -> None:
        """Plot radar data in real-time."""
        try:
            plt.ion()

            fig, ax = plt.subplots()
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([0, 0.6])
            ax.grid(True)

            self.mmwave.clear_frame_buffer()

            scatter = ax.scatter([], [], c="b", marker="o")

            while plt.fignum_exists(fig.number):
                frame = self.mmwave.get_frame()
                frame_tlv = self.mmwave.parse_tlv(frame)

                if frame_tlv["tlv_x"]:
                    scatter.set_offsets(np.c_[frame_tlv["tlv_x"], frame_tlv["tlv_y"]])

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.01)

        except Exception as e:
            logger.error(f"Error during plotting: {str(e)}")
        finally:
            plt.close("all")
            plt.ioff()

    def do_record(self, args: str) -> None:
        """Record gesture data.

        Args:
            args: "<Gesture name> <Recording times>"
        """
        if not args:
            logger.error("Please enter the gesture name and recording times like 'record left 10'")
            return

        name, times = args.split(" ")
        data_dir = Path("records")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Clear buffer to avoid affecting recording
        self.mmwave.clear_frame_buffer()

        for t in track(range(int(times)), description="[cyan]Recording data...[/cyan]"):
            buffer = self._collect_gesture_data()

            if len(buffer) < 4:
                logger.warning("[yellow]Insufficient data...[/yellow]")
                t -= 1
                continue

            now = datetime.datetime.now().strftime("%m%d%H%M%S")
            filename = f"{name}_{now}.npy"
            logger.info(f"Saving data {filename}")
            logger.debug(f"Data content {buffer}")

            np.save(data_dir / filename, buffer)

    def _collect_gesture_data(self) -> list:
        """Collect gesture data.

        Collects the gesture movement trajectory data detected by the radar.
        Data points are filtered when no movement is detected or the movement
        amplitude is too large.

        Returns:
            list: List of collected data points, each point contains [x, y] coordinates
        """
        flag = GESTURE_COLLECTION["INITIAL_FLAG"]
        buffer = []
        prev_x = None
        prev_y = None

        while True:
            frame = self.mmwave.get_frame()
            frame_tlv = self.mmwave.parse_tlv(frame)

            x, y = frame_tlv["tlv_x"], frame_tlv["tlv_y"]

            if not x:
                if flag == GESTURE_COLLECTION["END_FLAG"]:
                    break
                flag -= 1
                continue
            flag = GESTURE_COLLECTION["RESET_FLAG"]

            # Filter out data with too large movement amplitude
            if prev_x is not None and (abs(x - prev_x) > GESTURE_COLLECTION["POSITION_THRESHOLD"]).any():
                continue

            if prev_y is not None and (abs(y - prev_y) > GESTURE_COLLECTION["POSITION_THRESHOLD"]).any():
                continue

            # Calculate average position and round to the nearest integer
            x = round(np.mean(x), 3)
            y = round(np.mean(y), 3)

            logger.info(f"[blue]x {x} y {y}[/blue]")
            buffer.append([x, y])
            prev_x, prev_y = x, y

        return buffer

    def do_predict(self, args: str) -> None:
        """Predict gesture.

        Args:
            args: Model type, 'LSTM' or 'Conv2D', defaults to 'LSTM'
        """
        args = args or "LSTM"

        try:
            self._validate_args(args, ["LSTM", "Conv2D"])

            # Load model
            model_path = Path("models") / f"{args}.keras"
            if not model_path.exists():
                logger.error(f"[red]Model not found: {args}.keras")
                return

            model = load_model(model_path)
            logger.info(f"[green]Model loaded: {args}.keras")

            # Collect gesture data
            logger.info("Observing gesture...")
            buffer = self._collect_gesture_data()

            if not buffer:
                logger.error("[red]No gesture detected")
                return

            logger.info("Predicting gesture...")

            # Prepare data
            sequence_length = MODEL_CONFIG[args]["sequence_length"]
            gesture_data = np.zeros((1, sequence_length, 2))
            gesture_data[0, : min(len(buffer), sequence_length)] = buffer[:sequence_length]

            if args == "Conv2D":
                gesture_data = gesture_data.reshape((-1, sequence_length, 2, 1))

            # Make prediction
            prediction = model.predict(gesture_data, verbose=0)

            # Get prediction result
            predicted_gesture = np.argmax(prediction)
            gesture_label = GESTURE_MAP.get(predicted_gesture, "unknown")
            console.print(f"Probability: {prediction.max():.3f}", style="green")
            console.print(f"Prediction: {gesture_label}", style="green")

        except Exception as e:
            logger.error(str(e))

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (training data, labels)

        Raises:
            ValueError: Raised when no valid training data is found
        """
        data = []
        labels = []
        sequence_length = MODEL_CONFIG["LSTM"]["sequence_length"]

        records_dir = Path("records")
        if not records_dir.exists():
            raise ValueError("Training data directory not found")

        for filename in records_dir.glob("*.npy"):
            label = filename.stem.split("_")[0]
            if label not in GESTURES:
                logger.warning(f"Skipping file with unknown label: {filename}")
                continue

            content = np.load(filename)

            # Ensure data points are 2D
            if content.shape[1] != 2:
                logger.warning(f"Skipping file with incorrect shape: {filename}")
                continue

            # Pad or truncate data to specified length
            padded_content = np.zeros((sequence_length, 2))
            padded_content[: min(len(content), sequence_length)] = content[:sequence_length]

            data.append(padded_content)
            labels.append(GESTURES[label])
            logger.info(f"Loaded file: {filename}")

        if not data:
            raise ValueError("No valid training data found")

        return np.array(data), np.array(labels)

    def _create_model(self, model_type: str, input_shape: tuple, num_classes: int) -> Sequential:
        """Create model.

        Args:
            model_type: Model type ('LSTM' or 'Conv2D')
            input_shape: Input data shape
            num_classes: Number of classes

        Returns:
            Sequential: Created model
        """
        if model_type == "LSTM":
            return Sequential(
                [
                    LSTM(128, input_shape=input_shape),
                    Dense(num_classes, activation="softmax"),
                ]
            )
        else:  # Conv2D
            return Sequential(
                [
                    Conv2D(32, kernel_size=(3, 1), activation="relu", input_shape=input_shape),
                    Flatten(),
                    Dense(num_classes, activation="softmax"),
                ]
            )

    def do_train(self, args: str) -> None:
        """Train gesture recognition model.

        Args:
            args: Model type, 'LSTM' or 'Conv2D', defaults to 'LSTM'
        """
        args = args or "LSTM"

        try:
            self._validate_args(args, ["LSTM", "Conv2D"])
            logger.info("[cyan]Training model...[/cyan]")

            # Prepare training data
            data, labels = self._prepare_training_data()

            # Convert labels to one-hot encoding
            labels = to_categorical(labels)

            # Adjust data shape based on model type
            if args == "Conv2D":
                data = data.reshape((-1, data.shape[1], 2, 1))
                input_shape = (data.shape[1], 2, 1)
            else:  # LSTM
                input_shape = (data.shape[1], 2)

            # Create and compile model
            model = self._create_model(args, input_shape, labels.shape[1])
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            logger.info(f"Start training {args} model")

            # Train model
            model.fit(
                data,
                labels,
                epochs=MODEL_CONFIG[args]["epochs"],
                batch_size=MODEL_CONFIG[args]["batch_size"],
                validation_split=MODEL_CONFIG[args]["validation_split"],
                verbose=1,
            )

            # Save model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model.save(models_dir / f"{args}.keras")

            logger.info(f"[green]Training completed: {args}.keras")

        except Exception as e:
            logger.error(str(e))

    def do_clear(self, args: str) -> None:
        """Clear terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def do_exit(self, args: str) -> bool:
        """Exit the program."""
        logger.info("[cyan]Exiting the program...[/cyan]")
        return True
