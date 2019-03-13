import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

#print features(petal_len, ...) & type of flowers(setosa, ...)
print iris.feature_names
print iris.target_names

#print a dataset as per index '[0]'
print iris.data[0]
print iris.target[0]

# going to splitup the dataset (one for trainig & one for testing)
# we keep testing data seperate form training data later we use testing data to
#       test the accuracy of the classifier on the data it never seen before

# remove one example from each flower
test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#training classifier
clf = tree.DecisionTreeClassifier()
#assigning data into classifier
clf.fit(train_data, train_target)

# original targets
print test_target

#predicted targets
print clf.predict(test_data)
