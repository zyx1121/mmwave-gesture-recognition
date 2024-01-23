import os
import cmd
import colorama
import matplotlib.pyplot as plt
import numpy as np
import datetime

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
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

    # plt.ion()
    # plt.subplots()

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

        if prev_y is not None and (abs(y - prev_y) > 0.25).any():
          continue

        x = round(np.mean(x, axis=0), 3)
        y = round(np.mean(y, axis=0), 3)
        print(f'x:{x} y:{y}')
        buffer.append([x, y])
        prev_x = x

        # plt.clf()
        # plt.xlim(-0.5, 0.5)
        # plt.ylim(0.0, 1.0)

        # plt.plot(x, y, 'o', color='red')
        # plt.pause(0.005)

      # 資料不足 無法儲存
      if len(buffer) < 4:
        print(f'{Fore.RED}資料不足...')
        t -= 1
        continue

      now = datetime.datetime.now().strftime('%m%d%H%M%S')
      print(f'\n{Fore.GREEN}儲存資料... \n{buffer} \n{name}_{now}.npy\n')

      data = os.path.join(data_dir, f'{name}_{now}.npy')

      np.save(data, buffer)

  def do_predict(self, args):
    model = load_model('model.keras')

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

      print(f'{Fore.CYAN}預測手勢...')

      # 資料不足 無法預測
      if len(buffer) < 4:
        print(f'{Fore.RED}無法辨識')
        continue

      gesture_data = pad_sequences([buffer], maxlen=32, dtype='float32')

      # 進行預測
      prediction = model.predict(gesture_data)

      # 機率大於 0.9 才顯示結果
      if prediction.max() < 0.9:
        print(f'{Fore.RED}無法辨識')
        continue

      # 找出預測結果中機率最高的手勢
      predicted_gesture = np.argmax(prediction)

      # 將預測結果與手勢對應起來
      gestures = ['左揮', '右揮', '上揮', '下揮', '順時針', '逆時針']
      # 顯示預測結果機率
      print(f'{Fore.GREEN}手勢機率：{prediction.max()}')
      print(f'{Fore.GREEN}預測結果：{gestures[predicted_gesture]}')

  def do_train(self, args):
    print(f'\n{Fore.CYAN}訓練模型...\n')

    # 定義標籤到數字的映射
    label_to_num = {'left': 0, 'right': 1, 'up': 2, 'down': 3, 'cw': 4, 'ccw': 5}

    # 初始化資料和標籤的列表
    data = []
    labels = []

    # 列出所有檔案
    for filename in os.listdir('records'):
      # 獲取標籤和時間
      label, _, _ = filename.partition('_')

      # 讀取檔案內容
      content = np.load(os.path.join('records', filename))

      # 將資料和標籤加入到列表中
      data.append(content)
      labels.append(label_to_num[label])

    # 將資料裁剪或填充到相同的長度
    data = pad_sequences(data, maxlen=32, dtype='float32')

    # 將資料和標籤轉換成 numpy 陣列
    x_train = np.array(data)
    y_train = np.array(labels)

    # 儲存資料和標籤
    # np.save('x_train.npy', x_train)
    # np.save('y_train.npy', y_train)

    # 讀取資料和標籤
    # x_train = np.load('x_train.npy')
    # y_train = np.load('y_train.npy')

    # 將標籤轉換成 one-hot 編碼
    y_train = to_categorical(y_train)

    # 建立模型
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # 編譯模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 訓練模型
    model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

    model.save('test.keras')

    print(f'\n{Fore.GREEN}訓練完成 model.keras\n')


  def do_exit(self, args):
    return True

if __name__ == '__main__':
    Console().cmdloop()
