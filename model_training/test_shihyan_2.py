# 於命令列安裝套件:
# pip3 install git+https://github.com/keras-team/keras-tuner.git
# pip3 install autokeras matplotlib pydot pydot-ng pydotplus graphviz

# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import autokeras as ak

# ------------------------------------------------------------------------------

# 資料集來源：
# https://www.kaggle.com/nisargchodavadiya/imdb-movie-reviews-with-ratings-50k?select=imdb_sup.csv

df = pd.read_csv('https://github.com/alankrantas/IMDB-movie-reviews-with-ratings_dataset/raw/main/imdb_sup.csv')
print(df)

# ------------------------------------------------------------------------------
'''
x = df['Review'].to_numpy()
y = df['Rating'].to_numpy()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------

fig = plt.figure()
bin = np.arange(11) + 1

ax = fig.add_subplot(1, 2, 1)
ax.set_xticks(bin)
plt.hist(y_train, bins=bin-0.5, rwidth=0.9)
ax.set_title('Train dataset histogram')

ax = fig.add_subplot(1, 2, 2)
ax.set_xticks(bin)
plt.hist(y_test, bins=bin-0.5, rwidth=0.9)
ax.set_title('Test dataset histogram')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------

'''
reg = ak.TextRegressor(max_trials=15)
'''

cbs = [
    tf.keras.callbacks.EarlyStopping(patience=5)
]

input_node = ak.TextInput()
output_node = ak.TextBlock(block_type='ngram', max_tokens=30000)(input_node)
output_node = ak.RegressionHead()(output_node)

reg = ak.AutoModel(inputs=input_node, outputs=output_node,epochs=100, max_trials=15)
reg.fit(x_train, y_train, callbacks=cbs)

# ------------------------------------------------------------------------------

print(reg.evaluate(x_test, y_test))

# ------------------------------------------------------------------------------

predicted = reg.predict(x_test[:10]).flatten()

for i in range(10):
    print('Review:', x_test[i][:100], '...')
    print('Predict:', predicted[i].round(3))
    print('Real:', y_test[i])
    print('')

# ------------------------------------------------------------------------------

model = reg.export_model()
model.summary()

# ------------------------------------------------------------------------------

from tensorflow.keras.utils import plot_model
plot_model(model)
'''