import sklearn
import autokeras as ak
import logging
import datetime
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report

log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
logging.basicConfig(level=logging.DEBUG, filename=log_filename, filemode='w',
	#format='[%(levelname).1s %(asctime)s] %(message)s',
	format='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)
# load datasets
(data_train, target_train), (data_test, target_test) = \
             fashion_mnist.load_data()
# classifier
clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# training
clf.fit(data_train, target_train)
# making predictions on test data
predictions = clf.predict(data_test).astype('int8')
# print out classification results
print(classification_report(target_test, predictions))