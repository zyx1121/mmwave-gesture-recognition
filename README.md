# 毫米波雷達 AI 手勢辨識

###### Gesture Recognition Using mmWave Sensor - TI AWR1642

提供了 `設置`、`紀錄`、`訓練` 與 `預測` 功能，設定並收集 AWR1642 的毫米波雷達資料，即可辨識特定手勢等：
- 上揮
- 下揮
- 左揮
- 右揮
- 順時針旋轉
- 逆時針旋轉

> 也可以透過 `紀錄` 功能錄製手勢並訓練。

## 前置作業

應在 AWR1642 開發板上燒入官方提供的 Demo 程式
- [AWR1642BOOST 初始設置](https://gist.github.com/zyx1121/0756055fa9138aec81617501e2e5f263)

## 開始

- 下載並進入專案資料夾
  - `git clone https://github.com/zyx1121/mmwave.gesture.recognition`
  - `cd mmwave.gesture.recognition`

- 安裝依賴庫
  - `pip install -r requirements.txt`

- 進入 console 即可開始
  - `python console.py`

## 專案結構

- `mmwave.gesture.recognition/`
  
  - `models/`
    - `Conv2D.keras`
    - `LSTM.keras`
      
  - `records/`
    - `[label]_[%m%d%H%M%S]`
      
  - `console.py`
  - `mmwave.py`
  - `profile.cfg`

## 指令功能

- `cfg` : 將 `profile.cfg` 設定傳送到板子。

- `record [gesture] [times]` : 錄製 `[gesture]` 資料 `[times]` 次儲存至 `records/[gesture]_[date].npy`。

- `train [model]` : 訓練模型，預模型可選擇 `Conv2D` 或 `LSTM`。

- `predict [model]` : 實時抓取雷達資料丟進模型預測手勢，可選擇 `Conv2D` 或 `LSTM`。

- `exit` : 退出控制台。
