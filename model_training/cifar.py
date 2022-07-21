import os
os.system("pip3 install matplotlib")
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import autokeras as ak
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_test.shape)

# fig = plt.figure()
# bin = np.arange(11)

# ax = fig.add_subplot(1,2,1)
# ax.set_xticks(bin)
# plt.hist(y_train,bins=bin-0.5,rwidth=0.9)
# ax.set_title('Train dataset histogram')

# ax = fig.add_subplot(1,2,2)
# ax.set_xticks(bin)
# plt.hist(y_train,bins=bin-0.5,rwidth=0.9)
# ax.set_title('Train dataset histogram')
# plt.show()

clf = ak.ImageClassifier(max_trials=1,overwrite=True)
clf.fit(x_train,y_train,epochs=10,batch_size=2)
clf.evaluate(x_test,y_test)
# predicted=clf.predict(x_test[:10])
# labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# fig=plt.figure(figsize=(16,6))
# for i in range(10):
#     ax=fig.add_subplot(2,5,i+1)
#     ax.set_axis_off()
#     plt.imshow(x_test[i])
#     ax.set_title(f'Predicted:{labels[int(predicted[i])]},Real:{labels[int(y_test[i])]}')
#     plt.tight_layout()
#     plt.show()
model = clf.export_model()
model.summary()
# ------------------------------------------------------------------------------

# from tensorflow.keras.utils import plot_model
# plot_model(model)

# ------------------------------------------------------------------------------

model.save('cifar_best_model')

# ------------------------------------------------------------------------------

from tensorflow.keras.models import load_model

keras_model = load_model('cifar_best_model', custom_objects=ak.CUSTOM_OBJECTS)
print(keras_model.evaluate(x_test, y_test))

# ------------------------------------------------------------------------------

predicted = keras_model.predict(x_test)
predicted_classes = predicted.argmax(axis=1)

# ------------------------------------------------------------------------------

from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_classes, target_names=labels))