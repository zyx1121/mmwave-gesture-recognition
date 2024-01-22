import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences

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
print(x_train)
y_train = np.array(labels)
print(y_train)

# 儲存資料和標籤
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)