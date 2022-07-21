import tensorflow as tf
mnist = tf.keras.datasets.mnist

# 匯入 MNIST 手寫阿拉伯數字 訓練資料
(x_train, y_train),(x_test, y_test) = mnist.load_data()

import autokeras as ak

# 初始化影像分類器(image classifier)
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1)
# 訓練模型
clf.fit(x_train, y_train, epochs=10)


#####################
# 預測
predicted_y = model.predict(x_test)

# 評估，打分數
print(model.evaluate(x_test, y_test))
# 比較 20 筆
print('prediction:', ' '.join(predicted_y[0:20].ravel()))
print('actual    :', ' '.join(y_test[0:20].astype(str)))
# 顯示錯誤的資料圖像
import matplotlib.pyplot as plt

X2 = x_test[8,:,:]
plt.imshow(X2.reshape(28,28))
plt.show() 
# 使用小畫家，寫0~9，實際測試看看
from skimage import io
from skimage.transform import resize
import numpy as np

X_ALL = np.empty((0, 28, 28))
for i in range(10): 
    image1 = io.imread(f'./myDigits/{i}.png', as_gray=True)
    #image1 = Image.open(uploaded_file).convert('LA')
    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    X1 = image_resized.reshape(1, 28, 28) #/ 255
    # 反轉顏色
    # 顏色0為白色，與RGB顏色不同，(0,0,0) 為黑色。
    # 還原特徵縮放
    X1 = (np.abs(1-X1) * 255).astype(int)
    X_ALL = np.concatenate([X_ALL, X1])
predictions = model.predict(X_ALL)
print(predictions)