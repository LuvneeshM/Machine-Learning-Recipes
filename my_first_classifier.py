'''
My own classifier from scratch :D
->Simplified version of KNearestNeighbors

Building on previous code from pipeline.py
Pros:
-relatively simple to understand + replicate
Cons:
-Computation heavy since we iterate over every training point to make a prediction
-Hard to represent relationship between features
'''

from scipy.spatial import distance #used for euclidean distance

#a = pt from training data, b = pt from testing data
def euc(a,b): 
	#think of this like pythagorean theorem, but applies for all levels of dimensions
	return distance.euclidean(a,b)

#K-Nearest Neighbor Classifier
#Hard-code K = 1 for this one
class ScrappyKNN():
	def fit(self, x_train, y_train):
		#memorize the training data, x_train contains the featues
		self.X_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self,row):
		best_dist = euc (row, self.X_train[0])
		best_index = 0
		for i in range(1,len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i

		return self.y_train[best_index]


#Random Classifier
import random
class RandomClassifier():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = random.choice(self.y_train)
			predictions.append(label)
		return predictions


from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

#cross_validation is deprecated
from sklearn.model_selection import train_test_split
#y_test = true values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5)
'''
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

#different classifier that accomplishes the same thing as above one 
#Many different kinds of classifiers, high level they have same interface
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

^will be replaced by my own classifier, similiar to nearest neighbor classifier
-we will classify new points based on how close to its nearest neighbor (straight line distance == euclidean distance)
-if tie in distance:
	-can use random to pick one 
	-we can consider more neighbors using k (# of neighbors to consider)
'''


rand_classifier = RandomClassifier()
rand_classifier.fit(x_train,y_train)
rand_predictions = rand_classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print ("Accuracy of random classifier predictions is", accuracy_score(y_test, rand_predictions))

my_classifier = ScrappyKNN()
my_classifier.fit(x_train,y_train)
predictions = my_classifier.predict(x_test)


print ("Accuracy of my classifier predictions is", accuracy_score(y_test, predictions))