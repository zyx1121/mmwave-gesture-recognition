"""mmWave radar module.

This module provides functionality for communicating with and processing data from
the TI AWR1642 millimeter-wave radar.
"""

import re
import time
import struct
import codecs
import binascii
import math
from typing import Dict, List, Optional, Union
from pathlib import Path

from serial.tools import list_ports

from ..mmwave.uart import UART
from ..utils.logger import logger


def get_uint32(data: bytes) -> int:
    """Convert 4-byte data to unsigned 32-bit integer.

    Args:
        data: 4-byte data

    Returns:
        int: Converted integer
    """
    return data[0] + data[1] * 256 + data[2] * 65536 + data[3] * 16777216


class MMWave:
    """mmWave radar control class.

    Provides radar configuration, data acquisition and parsing functionality.

    Attributes:
        cli_port (UART): Command interface serial port
        data_port (UART): Data interface serial port
        config_file (Optional[Path]): Currently used configuration file
    """

    def __init__(
        self,
        cli_port: str,
        data_port: str,
        cli_rate: int = 115200,
        data_rate: int = 921600,
    ) -> None:
        """Initialize mmWave radar.

        Args:
            cli_port: Command interface port name
            data_port: Data interface port name
            cli_rate: Command interface baud rate
            data_rate: Data interface baud rate
        """
        self.cli_port = UART(cli_port, cli_rate)
        self.data_port = UART(data_port, data_rate)
        self.config_file: Optional[Path] = None

    @staticmethod
    def find_ports(pattern: str = "XDS110") -> List[str]:
        """Find serial ports matching the specified pattern.

        Args:
            pattern: Port name pattern to match

        Returns:
            List[str]: List of matching port names
        """
        ports = [str(p).split()[0] for p in list_ports.comports() if pattern in str(p)]
        ports.sort()
        return ports

    def connect(self) -> None:
        """Establish a connection with the radar."""
        self.cli_port.connect()
        self.data_port.connect()
        time.sleep(0.5)
        self._get_cmd()

    def send_configure(self, config_file: Union[str, Path]) -> bool:
        """Send configuration file to the radar.

        Args:
            config_file: Path to the configuration file

        Returns:
            bool: Whether the configuration is successful
        """
        config_path = Path(config_file)
        if not config_path.exists():
            logger.error(f"Configuration file does not exist: {config_file}")
            return False

        with open(config_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Skip empty lines and comment lines
            if re.match(r"(^\s*%|^\s*$)", line):
                continue

            logger.info(f"Sending command {line.strip()}")
            response = self._send_cmd(line)
            if response is None:
                self.config_file = None
                logger.error("Command sending failed")
                return False

            response = self._get_cmd()

            if "Done" in response:
                logger.info(f"Command executed {response.strip()}")
            elif "Ignored" in response or "Debug:" in response:
                logger.warning(f"Command ignored {response.strip()}")
            else:
                logger.error(f"Command failed {response.strip()}")
                self.reset()
                self.config_file = None
                return False

            if "sensorStart" not in line:
                # Wait for the radar to start
                time.sleep(0.01)

        self.config_file = config_path
        return True

    def _send_cmd(self, cmd: str) -> Optional[bytes]:
        """Send command to the radar.

        Args:
            cmd: Command to send

        Returns:
            Optional[bytes]: Command response
        """
        return self.cli_port.write(cmd.encode())

    def _get_cmd(self) -> str:
        """Get command response.

        Returns:
            str: Command response string
        """
        first_byte = self.cli_port.read(1)

        if first_byte:
            self.cli_port.port.timeout = 0.05

        response = first_byte + b"".join(iter(lambda: self.cli_port.read(1), b""))
        if not response:
            return ""

        self.cli_port.port.timeout = 0.5
        return response.decode(errors="ignore").replace("mmwDemo:/>", "")

    def reset(self) -> None:
        """Reset radar connection."""
        self.cli_port.reset()
        self.data_port.reset()

    def clear_frame_buffer(self) -> None:
        """Clear frame data buffer."""
        self.data_port.reset()
        self.data_port.clear()

    def get_frame(self) -> bytes:
        """Get one frame of radar data.

        Returns:
            bytes: Raw frame data
        """
        MAGIC_NUMBER = b"\x02\x01\x04\x03\x06\x05\x08\x07"

        frame = b""
        while frame.find(MAGIC_NUMBER) == -1:
            frame += self.data_port.read(1)

        if not frame.startswith(MAGIC_NUMBER):
            frame = frame[frame.find(MAGIC_NUMBER) :]

        frame += self.data_port.read(32)
        header = self.parse_header(frame)
        frame += self.data_port.read(header["packet_len"] - 40)

        return frame

    def parse_header(self, frame: bytes) -> Dict[str, Union[bytes, int]]:
        """Parse frame header data.

        Args:
            frame: Raw frame data

        Returns:
            Dict[str, Union[bytes, int]]: Parsed frame header information
        """
        HEADER_FORMAT = {
            "magic": "8s",
            "version": "4s",
            "packet_len": "I",
            "platform": "4s",
            "frame_num": "I",
            "time_cpu_cyc": "I",
            "num_det_obj": "I",
            "num_tlvs": "I",
            "unknown": "I",
        }

        header = struct.unpack_from("<" + "".join(HEADER_FORMAT.values()), frame)
        return dict(zip(HEADER_FORMAT.keys(), header))

    def parse_tlv(self, frame: bytes) -> Dict[str, List[float]]:
        """Parse TLV (Type-Length-Value) data.

        Args:
            frame: Raw frame data

        Returns:
            Dict[str, List[float]]: Parsed TLV data, including x and y coordinate lists
        """
        tlv_start = 40
        tlv_type = get_uint32(frame[tlv_start : tlv_start + 4])
        tlv_length = get_uint32(frame[tlv_start + 4 : tlv_start + 8])

        tlv_x: List[float] = []
        tlv_y: List[float] = []
        tlv_z: List[float] = []
        tlv_v: List[float] = []
        tlv_range: List[float] = []
        tlv_azimuth: List[float] = []
        tlv_elevation: List[float] = []

        header = self.parse_header(frame)
        offset = 8

        if tlv_type == 1:
            for obj in range(header["num_det_obj"]):
                # Parse x, y, z, v data
                x = struct.unpack(
                    "<f",
                    codecs.decode(
                        binascii.hexlify(frame[tlv_start + offset : tlv_start + offset + 4]),
                        "hex",
                    ),
                )[0]
                y = struct.unpack(
                    "<f",
                    codecs.decode(
                        binascii.hexlify(frame[tlv_start + offset + 4 : tlv_start + offset + 8]),
                        "hex",
                    ),
                )[0]
                z = struct.unpack(
                    "<f",
                    codecs.decode(
                        binascii.hexlify(frame[tlv_start + offset + 8 : tlv_start + offset + 12]),
                        "hex",
                    ),
                )[0]
                v = struct.unpack(
                    "<f",
                    codecs.decode(
                        binascii.hexlify(frame[tlv_start + offset + 12 : tlv_start + offset + 16]),
                        "hex",
                    ),
                )[0]

                # Calculate polar coordinate parameters
                detected_range = math.sqrt(x**2 + y**2 + z**2)

                if y == 0:
                    detected_azimuth = 90 if x >= 0 else -90
                else:
                    detected_azimuth = math.degrees(math.atan(x / y))

                if x == 0 and y == 0:
                    detected_elevation = 90 if z >= 0 else -90
                else:
                    detected_elevation = math.degrees(math.atan(z / math.sqrt(x**2 + y**2)))

                # Add to the result list
                tlv_x.append(round(x, 3))
                tlv_y.append(round(y, 3))
                tlv_z.append(z)
                tlv_v.append(v)
                tlv_range.append(detected_range)
                tlv_azimuth.append(detected_azimuth)
                tlv_elevation.append(detected_elevation)

                offset += 16

        return {
            "tlv_x": tlv_x,
            "tlv_y": tlv_y,
            "tlv_z": tlv_z,
            "tlv_v": tlv_v,
            "tlv_range": tlv_range,
            "tlv_azimuth": tlv_azimuth,
            "tlv_elevation": tlv_elevation,
        }
