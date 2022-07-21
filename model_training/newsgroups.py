# 於命令列安裝套件:
# pip3 install git+https://github.com/keras-team/keras-tuner.git
# pip3 install autokeras matplotlib pydot pydot-ng pydotplus graphviz

# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import autokeras as ak

# ------------------------------------------------------------------------------

from sklearn.datasets import fetch_20newsgroups

train = fetch_20newsgroups(subset='train')
print(train.target_names)

# ------------------------------------------------------------------------------

categories = ['comp.sys.ibm.pc.hardware',
              'rec.autos',
              'rec.sport.baseball', 
              'sci.med',
              'sci.space',
              'talk.politics.mideast']

train = fetch_20newsgroups(subset='train', 
                           categories=categories,
                           remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test',
                          categories=categories,
                          remove=('headers', 'footers', 'quotes'))

x_train = np.array(train.data)
y_train = np.array(train.target)
x_test = np.array(test.data)
y_test = np.array(test.target)

print(x_train.shape)
print(x_test.shape)

# ------------------------------------------------------------------------------

fig = plt.figure()
bin = np.arange(len(categories) + 1)

labels = ('PC hardware', 'Automobile', 'Baseball', 
          'Medicine', 'Space', 'Politics', '')

ax = fig.add_subplot(1, 2, 1)
ax.set_xticks(bin)
ax.set_xticklabels(labels, rotation=90)
plt.hist(y_train, bins=bin-0.5, rwidth=0.9)
ax.set_title('Train dataset histogram')

ax = fig.add_subplot(1, 2, 2)
ax.set_xticks(bin)
ax.set_xticklabels(labels, rotation=90)
plt.hist(y_test, bins=bin-0.5, rwidth=0.9)
ax.set_title('Test dataset histogram')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------

from keras_tuner.engine.hyperparameters import Choice

pretraining = Choice(name='pretraining', values=['word2vec'])
num_blocks = Choice(name='num_blocks', values=[1])

cbs = [tf.keras.callbacks.EarlyStopping(patience=3)]

input_node = ak.TextInput()

output_node = ak.TextToIntSequence(max_tokens=50000)(input_node)
output_node = ak.Embedding(pretraining=pretraining, max_features=50000)(output_node)
output_node = ak.ConvBlock(num_blocks=num_blocks, separable=True, max_pooling=True)(output_node)
output_node = ak.SpatialReduction(reduction_type='global_max')(output_node)
output_node = ak.ClassificationHead()(output_node)

clf = ak.AutoModel(inputs=input_node, outputs=output_node, 
                   max_trials=20, overwrite=True)
clf.fit(x_train, y_train, callbacks=cbs, batch_size=5)

# ------------------------------------------------------------------------------

print(clf.evaluate(x_test, y_test))

# ------------------------------------------------------------------------------

predicted = clf.predict(x_test).flatten().astype('uint8')

for i in range(10):
    print('TEXT [')
    print(x_test[i].strip()[:400])
    print(f'] PREDICTED: {labels[predicted[i]]}, REAL: {labels[y_test[i]]}')
    print('')

# ------------------------------------------------------------------------------

labels = ('PC hardware', 'Automobile', 'Baseball', 'Medicine', 'Space', 'Politics')

from sklearn.metrics import classification_report
print(classification_report(y_test, predicted, target_names=labels))

# ------------------------------------------------------------------------------

model = clf.export_model()
model.summary()

# ------------------------------------------------------------------------------

from tensorflow.keras.utils import plot_model
plot_model(model)