import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# 讀取資料和標籤
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# 將標籤轉換成 one-hot 編碼
y_train = to_categorical(y_train)

# 建立模型
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1], activation='softmax'))

# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

model.save('model.keras')

print('good')