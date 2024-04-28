import os
import cmd
import colorama
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from colorama import Fore
from mmwave import mmWave

colorama.init(autoreset=True)

mpl.use('Qt5Agg')
mpl.rcParams['toolbar'] = 'None'

class Console(cmd.Cmd):
  def __init__(self):
    super().__init__()

    os.system('cls' if os.name == 'nt' else 'clear')

    self.prompt = f'{Fore.BLUE}AWR1642 > {Fore.RESET}'
    self.exit = False

    self.mmwave_init()

  def mmwave_init(self):
    print(f'\n{Fore.CYAN}尋找裝置...')
    ports = mmWave.find_ports()

    if len(ports) < 2:
      print(f'{Fore.RED}找不到裝置.')
      exit(0)

    cli_port, data_port = ports[0], ports[1]

    self.mmwave = mmWave(ports[0], ports[1], cli_rate=115200, data_rate=921600)
    self.mmwave.connect()
    print(f'{Fore.GREEN}連線成功 CLI: {cli_port}  DATA: {data_port}\n')

  def validate_args(self, args, valid_args):
    if args not in valid_args:
      raise Exception(f'{Fore.RED}參數必須是 {valid_args}')

  def do_cfg(self, args=''):
    if args == '':
      args = 'profile'

    print(f'\n{Fore.CYAN}設定裝置...\n')
    mmwave_configured = self.mmwave.send_configure(f'{args}.cfg')
    if not mmwave_configured:
      return

  def do_plot(self, args):
    self.mmwave.clear_frame_buffer()

    plt.ion()
    plt.subplots()
    plt.xlim(-0.5, 0.5)
    plt.ylim(0.0, 1.0)
    plt.scatter([None], [None], s=20, c='red')
    plt.grid(True)

    count = 0
    is_quit = {'status': False}

    def on_key(event):
      if event.key == 'q':
        is_quit['status'] = True
        plt.close()

    plt.connect('key_press_event', on_key)

    while True:
      if is_quit['status']:
        print(f'{Fore.RED}用戶退出\n')
        return

      frame = self.mmwave.get_frame()
      tlv = self.mmwave.parse_tlv(frame)

      x, y = tlv['tlv_x'], tlv['tlv_y']

      plt.clf()
      plt.xlim(-0.5, 0.5)
      plt.ylim(0.0, 1.0)
      plt.scatter(x, y, s=20, c='red')
      plt.grid(True)
      plt.pause(0.005)

      print(f'x:{x} y:{y}')

      if not x:
        count += 1
        if count > 30:
          plt.close()
          print(f'{Fore.RED}超時自動退出\n')
          return
        continue

      count = 0

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

        if prev_x is not None and (abs(x - prev_x) > 0.3).any():  # Compare current x with previous x
          continue  # Discard the current x value

        if prev_y is not None and (abs(y - prev_y) > 0.3).any():
          continue

        x = round(np.mean(x, axis=0), 3)
        y = round(np.mean(y, axis=0), 3)
        print(f'x:{x} y:{y}')
        buffer.append([x, y])
        prev_x = x

      # 資料不足 無法儲存
      if len(buffer) < 4:
        print(f'{Fore.RED}資料不足...')
        t -= 1
        continue

      now = datetime.datetime.now().strftime('%m%d%H%M%S')
      print(f'\n{Fore.GREEN}儲存資料... \n{buffer} \n{name}_{now}.npy\n')

      data = os.path.join(data_dir, f'{name}_{now}.npy')

      np.save(data, buffer)

  def do_predict(self, args=''):
    if args == '':
      args = 'LSTM'

    self.validate_args(args, ['LSTM', 'Conv2D'])

    model = load_model(f'models/{args}.keras')

    self.mmwave.clear_frame_buffer()

    while True:
      print(f'\n{Fore.CYAN}觀察手勢...')
      count = 0
      flag = -1
      buffer = []
      prev_x = None  # Store the previous x value
      prev_y = None  # Store the previous y value
      while True:
        frame = self.mmwave.get_frame()
        frame_tlv = self.mmwave.parse_tlv(frame)

        x, y = frame_tlv['tlv_x'], frame_tlv['tlv_y']

        if not x:
          count += 1
          if count > 90:
            print(f'{Fore.RED}超時自動退出\n')
            return

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

  def do_train(self, args=''):
    if args == '':
      args = 'LSTM'

    self.validate_args(args, ['LSTM', 'Conv2D'])

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

      print(f'{Fore.GREEN}讀取檔案 {filename}')

    # 將資料裁剪或填充到相同的長度
    data = pad_sequences(data, maxlen=32, dtype='float32')

    # 將資料和標籤轉換成 numpy 陣列
    x_train = np.array(data)
    y_train = np.array(labels)

    # 將標籤轉換成 one-hot 編碼
    y_train = to_categorical(y_train)

    print(f'{Fore.GREEN}資料準備完成\n')

    # 建立 LSTM 模型
    if args == 'LSTM':
      model = Sequential()
      model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
      model.add(Dense(y_train.shape[1], activation='softmax'))

    # 建立 CNN 模型
    elif args == 'Conv2D':
      x_train = x_train.reshape((-1, 32, 2, 1))
      model = Sequential()
      model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(32, 2, 1)))
      model.add(Flatten())
      model.add(Dense(y_train.shape[1], activation='softmax'))

    print(f'{Fore.GREEN}建立模型 {args}\n')

    # 編譯模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f'{Fore.GREEN}開始訓練模型 {args}\n')

    # 訓練模型
    model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

    model.save(os.path.join('models', f'{args}.keras'))

    print(f'\n{Fore.GREEN}訓練完成 {args}.keras\n')

  def do_clear(self, args):
    os.system('cls' if os.name == 'nt' else 'clear')

  def do_exit(self, args):
    return True

if __name__ == '__main__':
  try:
    Console().cmdloop()
  except KeyboardInterrupt:
    print(f'\n{Fore.RED}用戶退出')
    exit(0)
