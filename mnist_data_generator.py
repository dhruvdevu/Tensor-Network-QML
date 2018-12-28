import numpy as np
import mnist_data_process
import sklearn.preprocessing as sp
gen = mnist_data_process.DataGenerator()
train_data, train_labels, test_data, test_labels = gen.load_data(classes=set([0, 1]))
train_data = sp.normalize(train_data)
test_data = sp.normalize(test_data)
print(len(train_data), len(test_data))
np.savetxt("data/train_data_mnist.csv", train_data, delimiter = ",")
np.savetxt("data/train_labels_mnist.csv", train_labels, delimiter = ",")
np.savetxt("data/test_data_mnist.csv", test_data, delimiter = ",")
np.savetxt("data/test_labels_mnist.csv", test_labels, delimiter = ",")
