import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import autokeras as ak

data_df = pd.read_csv('https://github.com/alankrantas/IMDB-movie-reviews-with-ratings_dataset/raw/main/imdb_sup.csv')
print(data_df)

x = df['Review'].to_numpy()
y = df['Rating'].to_numpy()

#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

cbs = [tf.keras.callbacks.EarlyStopping(patience=3)]
reg=ak.TextRegressor(max_trials=10)
reg.fit(x_train, y_train, callbacks=cbs)