import os
import cmd
import colorama
import matplotlib.pyplot as plt
import numpy as np
import datetime

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from colorama import Fore
from mmwave import mmWave

colorama.init(autoreset=True)

class Console(cmd.Cmd):
  def __init__(self):
    super().__init__()

    self.prompt = f'{Fore.BLUE}mmWave> {Fore.RESET}'
    self.mmwave_init()

  def mmwave_init(self):
    self.mmwave = None

    print(f'\n{Fore.CYAN}尋找裝置...')
    ports = mmWave.find_ports()

    if len(ports) < 2:
      print(f'{Fore.RED}找不到裝置.')
      return

    if len(ports) > 2:
      print(f'{Fore.YELLOW}找到多個裝置. 使用前兩個裝置: {ports[:2]}')
      ports = ports[:2]

    cli_port, data_port = ports[0], ports[1]

    self.mmwave = mmWave(cli_port, data_port, cli_rate=115200, data_rate=921600)
    self.mmwave.connect()
    print(f'{Fore.GREEN}連線成功 CLI: {cli_port}  DATA: {data_port}\n')

  def do_cfg(self, args=''):
    if args == '':
      args = 'profile'

    config_dir = os.path.join(os.path.dirname(__file__), 'profiles')
    config = os.path.join(config_dir, f'{args}.cfg')

    print(f'\n{Fore.CYAN}設定裝置...\n')
    mmwave_configured = self.mmwave.send_configure(config)
    if not mmwave_configured:
      return

  def do_test(self, args):
    print(f'\n{Fore.CYAN}監聽資料...\n')

    while True:
      frame = self.mmwave.get_frame()
      frame_tlv = self.mmwave.parse_tlv(frame)

      print(f'x:{frame_tlv["tlv_x"]} y:{frame_tlv["tlv_y"]}\n')

  def do_plot(self, args):
    print(f'\n{Fore.CYAN}繪製圖表...\n')

    plt.ion()
    plt.subplots()

    while True:
      frame = self.mmwave.get_frame()
      frame_tlv = self.mmwave.parse_tlv(frame)

      plt.clf()
      plt.xlim(-0.5, 0.5)
      plt.ylim(0.0, 1.0)

      x, y = frame_tlv['tlv_x'], frame_tlv['tlv_y']

      print(f'x:{x} y:{y}\n')

      plt.plot(x, y, 'o', color='red')

      plt.pause(0.005)

  def do_record(self, args=''):
    if args == '':
      return

    name, times = args.split(' ')

    data_dir = os.path.join(os.path.dirname(__file__), 'records')

    self.mmwave.clear_frame_buffer()

    for t in range(int(times)):
      print(f'\n{Fore.CYAN}錄製資料... {t+1}\n')
      flag = -1
      buffer = []
      prev_x = None  # Store the previous x value
      prev_y = None  # Store the previous y value
      while True:
        frame = self.mmwave.get_frame()
        frame_tlv = self.mmwave.parse_tlv(frame)

        x, y = frame_tlv['tlv_x'], frame_tlv['tlv_y']

        if frame_tlv['tlv_x'] == []:
          if flag == 0:
            break
          else:
            flag -= 1
          continue
        else:
          flag = 4

        if prev_x is not None and (abs(x - prev_x) > 0.25).any():  # Compare current x with previous x
          continue  # Discard the current x value

        if prev_y is not None and (abs(y - prev_y) > 0.28).any():
          continue

        x = round(np.mean(x, axis=0), 3)
        y = round(np.mean(y, axis=0), 3)
        print(f'x:{x} y:{y}')
        buffer.append([x, y])
        prev_x = x

        plt.clf()
        plt.xlim(-0.5, 0.5)
        plt.ylim(0.0, 1.0)

        plt.plot(x, y, 'o', color='red')
        plt.pause(0.005)

      # 資料不足 無法儲存
      if len(buffer) < 4:
        print(f'{Fore.RED}資料不足...')
        t -= 1
        continue

      now = datetime.datetime.now().strftime('%m%d%H%M%S')
      print(f'\n{Fore.CYAN}儲存資料... {buffer} 到 {name}_{now}.npy\n')

      data = os.path.join(data_dir, f'{name}_{now}.npy')

      np.save(data, buffer)

  def do_predict(self, args):
    model = load_model('model.keras')

    plt.ion()
    plt.subplots()

    while True:
      print(f'\n{Fore.CYAN}觀察手勢...')
      flag = -1
      buffer = []
      prev_x = None  # Store the previous x value
      prev_y = None  # Store the previous y value
      while True:
        frame = self.mmwave.get_frame()
        frame_tlv = self.mmwave.parse_tlv(frame)

        x, y = frame_tlv['tlv_x'], frame_tlv['tlv_y']

        if frame_tlv['tlv_x'] == []:
          if flag == 0:
            break
          else:
            flag -= 1
          continue
        else:
          flag = 4

        if prev_x is not None and (abs(x - prev_x) > 0.25).any():  # Compare current x with previous x
          continue  # Discard the current x value

        if prev_y is not None and (abs(y - prev_y) > 0.25).any():
          continue

        x = round(np.mean(x, axis=0), 3)
        y = round(np.mean(y, axis=0), 3)
        print(f'x:{x} y:{y}')
        buffer.append([x, y])
        prev_x = x

        plt.clf()
        plt.xlim(-0.5, 0.5)
        plt.ylim(0.0, 1.0)

        plt.plot(x, y, 'o', color='red')
        plt.pause(0.005)

      plt.clf()

      print(f'\n{Fore.CYAN}預測手勢...')

      # 資料不足 無法預測
      if len(buffer) < 4:
        print(f'{Fore.RED}無法辨識')
        continue

      gesture_data = pad_sequences([buffer], maxlen=32, dtype='float32')

      # 進行預測
      prediction = model.predict(gesture_data)

      # 顯示預測結果機率
      print('結果機率：', prediction)

      # 機率大於 0.9 才顯示結果
      if prediction.max() < 0.9:
        print(f'{Fore.RED}無法辨識')
        continue

      # 找出預測結果中機率最高的手勢
      predicted_gesture = np.argmax(prediction)

      # 將預測結果與手勢對應起來
      gestures = ['左揮', '右揮', '上揮', '下揮', '順時針', '逆時針']
      print(f'{Fore.GREEN}預測結果：{gestures[predicted_gesture]}')


  def do_exit(self, args):
    return True

if __name__ == '__main__':
    Console().cmdloop()
