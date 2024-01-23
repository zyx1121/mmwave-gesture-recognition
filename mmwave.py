import re
import time
import serial
import struct
import codecs
import binascii
import math
import colorama

from serial.tools import list_ports
from colorama import Fore

colorama.init(autoreset=True)

def getUint32(data):
  return (data[0] +
          data[1] * 256 +
          data[2] * 65536 +
          data[3] * 16777216)

class uart:
  def __init__(self, name, rate):
    self.name = name
    self.rate = rate
    self.port = None

  def connect(self):
    print(f'{Fore.BLUE}連線通道 {self.name}')
    self.port = serial.Serial(self.name,
                              bytesize=serial.EIGHTBITS,
                              parity=serial.PARITY_NONE,
                              stopbits=serial.STOPBITS_ONE,
                              xonxoff=False,
                              rtscts=False,
                              dsrdtr=False,
                              writeTimeout=0,
                              timeout=0.5)
    self.port.baudrate = self.rate

  def reset(self):
    self.port.close()
    self.port.open()

  def clear(self):
    self.port.reset_input_buffer()
    self.port.reset_output_buffer()

  def write(self, data):
    self.port.write(data)
    return self.port.readline()

  def read(self, size=None):
    if size is None:
      size = self.port.in_waiting
    return self.port.read(size)

  def readline(self):
    return self.port.readline()


class mmWave:
  def __init__(self, cli_port, data_port, cli_rate, data_rate):
    self.cli_port = uart(cli_port, cli_rate)
    self.data_port = uart(data_port, data_rate)
    self.config_file = None

  def find_ports(pattern='XDS110'):
    ports = [str(p).split()[0] for p in list_ports.comports()
      if pattern in str(p)]
    ports.sort()
    return ports

  def connect(self):
    self.cli_port.connect()
    self.data_port.connect()
    time.sleep(.5)
    self.get_cmd()

  def send_cmd(self, cmd):
    return self.cli_port.write(cmd.encode())

  def get_cmd(self):
    first_byte = self.cli_port.read(1)

    if first_byte:
      self.cli_port.port.timeout = 0.05

    response = first_byte + b''.join(iter(lambda: self.cli_port.read(1), b''))
    if not response:
      return None

    self.cli_port.port.timeout = 0.5
    return response.decode(errors='ignore')

  def send_configure(self, config_file):
    with open(config_file, 'r') as f:
      lines = f.readlines()

    for line in lines:

      # 跳過註解與空白行
      if re.match(r'(^\s*%|^\s*$)', line):
        continue

      print(f'傳送指令: {Fore.YELLOW}{line}', end='')
      response = self.send_cmd(line)
      if response is None:
        self.config_file = None
        return False

      response = self.get_cmd()
      response = response.replace('mmwDemo:/>', '')
      if 'Done' in response:
        print(f'收到回應: {Fore.GREEN}{response}')
      elif 'Ignored' in response or 'Debug:' in response:
        print(f'收到回應: {Fore.YELLOW}{response}')
      else:
        print(f'收到回應: {Fore.RED}{response}')
        print(f'{Fore.RED}設定失敗')
        self.reset()
        self.config_file = None
        return False

      if 'sensorStart' not in line:
        time.sleep(0.01)

    print(f'{Fore.GREEN}設定完成\n')

    self.config_file = config_file
    return True

  def clear_frame_buffer(self):
    self.data_port.reset()
    self.data_port.clear()

  def get_frame(self):
    MAGIC_NUMBER = b'\x02\x01\x04\x03\x06\x05\x08\x07'

    frame = b''

    while frame.find(MAGIC_NUMBER) == -1:
      frame += self.data_port.read(1)

    if not frame.startswith(MAGIC_NUMBER):
      frame = frame[frame.find(MAGIC_NUMBER):]

    frame += self.data_port.read(32)

    header = self.parse_header(frame)

    frame += self.data_port.read(header['packet_len'] - 40)

    return frame

  def parse_tlv(self, frame):
    tlv_start  = 40
    tlv_type   = getUint32(frame[tlv_start+0:tlv_start+4:1])
    tlv_length = getUint32(frame[tlv_start+4:tlv_start+8:1])

    tlv_x = []
    tlv_y = []
    tlv_z = []
    tlv_v = []
    tlv_range = []
    tlv_azimuth = []
    tlv_elevation = []

    header = self.parse_header(frame)

    offset = 8

    if tlv_type == 1:
      for obj in range(header['num_det_obj']):
        x = struct.unpack('<f', codecs.decode(binascii.hexlify(frame[tlv_start+offset    : tlv_start+offset+4  : 1]), 'hex'))[0]
        y = struct.unpack('<f', codecs.decode(binascii.hexlify(frame[tlv_start+offset+4  : tlv_start+offset+8  : 1]), 'hex'))[0]
        z = struct.unpack('<f', codecs.decode(binascii.hexlify(frame[tlv_start+offset+8  : tlv_start+offset+12 : 1]), 'hex'))[0]
        v = struct.unpack('<f', codecs.decode(binascii.hexlify(frame[tlv_start+offset+12 : tlv_start+offset+16 : 1]), 'hex'))[0]

        detected_range = math.sqrt(x**2 + y**2 + z**2)

        if y == 0:
          if x >= 0:
            detected_azimuth = 90
          else:
            detected_azimuth = -90
        else:
          detected_azimuth = math.degrees(math.atan(x / y))

        if x == 0 and y == 0:
          if z >= 0:
            detected_elevation = 90
          else:
            detected_elevation = -90
        else:
          detected_elevation = math.degrees(math.atan(z / math.sqrt(x**2 + y**2)))

        tlv_x.append(round(x, 3))
        tlv_y.append(round(y, 3))
        tlv_z.append(z)
        tlv_v.append(v)
        tlv_range.append(detected_range)
        tlv_azimuth.append(detected_azimuth)
        tlv_elevation.append(detected_elevation)

        offset += 16

    return {'tlv_x': tlv_x, 'tlv_y': tlv_y}

  def parse_header(self, frame):
    HEADER_FORMAT = {
      'magic': '8s',
      'version': '4s',
      'packet_len': 'I',
      'platform': '4s',
      'frame_num': 'I',
      'time_cpu_cyc': 'I',
      'num_det_obj': 'I',
      'num_tlvs': 'I',
      'unknown': 'I'
    }

    header = struct.unpack_from('<' + ''.join(HEADER_FORMAT.values()), frame)

    header = dict(zip(HEADER_FORMAT.keys(), header))

    return header
