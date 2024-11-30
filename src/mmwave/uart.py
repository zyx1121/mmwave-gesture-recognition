"""UART communication module.

This module provides functionality for communicating with serial devices.
"""

import serial
from typing import Optional, Union

from ..utils.logger import logger


class UART:
    """UART communication class.

    Provides basic serial communication functionality including connection,
    read/write operations and buffer control.

    Attributes:
        name (str): Port name
        rate (int): Baud rate
        port (Optional[serial.Serial]): Serial port object
    """

    def __init__(self, name: str, rate: int = 115200) -> None:
        """Initialize UART object.

        Args:
            name: Port name
            rate: Baud rate
        """
        self.name = name
        self.rate = rate
        self.port: Optional[serial.Serial] = None

    def connect(self) -> None:
        """Establish serial connection."""
        logger.info(f"Connecting to port {self.name}")
        self.port = serial.Serial(
            self.name,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            writeTimeout=0,
            timeout=0.5,
        )
        self.port.baudrate = self.rate

    def reset(self) -> None:
        """Reset serial connection."""
        if self.port:
            self.port.close()
            self.port.open()

    def clear(self) -> None:
        """Clear input/output buffers."""
        if self.port:
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()

    def write(self, data: Union[str, bytes]) -> bytes:
        """Write data to the serial port.

        Args:
            data: Data to write

        Returns:
            bytes: Read response data
        """
        if self.port:
            if isinstance(data, str):
                data = data.encode()
            self.port.write(data)
            return self.port.readline()
        return b""

    def read(self, size: Optional[int] = None) -> bytes:
        """Read data from the serial port.

        Args:
            size: Number of bytes to read, if None, read all available data

        Returns:
            bytes: Read data
        """
        if self.port:
            if size is None:
                size = self.port.in_waiting
            return self.port.read(size)
        return b""

    def readline(self) -> bytes:
        """Read a line of data.

        Returns:
            bytes: Read data line
        """
        if self.port:
            return self.port.readline()
        return b""
