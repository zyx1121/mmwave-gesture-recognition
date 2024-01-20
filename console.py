import os
import cmd
import colorama
import matplotlib.pyplot as plt

from colorama import Fore
from mmwave import mmWave

colorama.init(autoreset=True)

class Console(cmd.Cmd):
  def __init__(self):
    super().__init__()

    self.prompt = f'{Fore.BLUE}mmWave> {Fore.RESET}'

    self.config_dir = os.path.join(os.path.dirname(__file__), 'profiles')
    self.default_config = 'profile'
    self.config = None
    self.parser = None

    self.mmwave_init()

  def mmwave_init(self):
    self.mmwave = None

    print(f'\n{Fore.CYAN}尋找裝置...')
    ports = mmWave.find_ports()

    if len(ports) < 2:
      print(f'{Fore.RED}找不到裝置.')
      return True

    if len(ports) > 2:
      print(f'{Fore.YELLOW}找到多個裝置. 使用前兩個裝置: {ports[:2]}')
      ports = ports[:2]

    cli_port, data_port = ports[0], ports[1]

    self.mmwave = mmWave(cli_port, data_port, cli_rate=115200, data_rate=921600)
    self.mmwave.connect()
    print(f'{Fore.GREEN}連線成功 CLI: {cli_port}  DATA: {data_port}\n')

  def do_cfg(self, args=''):
    if args == '':
      args = self.default_config

    config = os.path.join(self.config_dir, f'{args}.cfg')

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
      plt.show()

      plt.pause(0.005)

  def do_exit(self, args):
    return True

if __name__ == '__main__':
    Console().cmdloop()
