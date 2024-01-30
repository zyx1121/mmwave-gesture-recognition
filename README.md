# 毫米波雷達 AI 手勢辨識

Gesture Recognition Using mmWave Sensor - TI AWR1642
透過設定並收集 AWR1642 的雷達資料，透過 LSTM 模型辨識特定手勢：
- 上揮
- 下揮
- 左揮
- 右揮
- 順時針旋轉
- 逆時針旋轉

<br>

## 前置作業

- [AWR1642BOOST 初始設置](https://gist.github.com/zyx1121/0756055fa9138aec81617501e2e5f263)

<br>

## 指令功能

- `cfg` - 將 `profiles/profile.cfg` 設定傳送到板子
- `record [gesture] [times]` - 錄製 `[gesture]` 資料 `[times]` 次儲存至 `records/[gesture]_[date].npy`
- `train` - 訓練模型
- `predict` - 實時抓取雷達資料丟進模型預測手勢
- `exit` - 退出程式
