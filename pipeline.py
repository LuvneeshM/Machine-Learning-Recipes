'''
Pipelines for supervised learning
-We dont want to write our own function used to classify feattures (as we are subjective and cant account all possibilities)
-We want our algorithm to learn it from training data
	-To learn a function (takes input and returns output)
	-best to start with a model that can be adjusted with train data
		-We can start with a random model and then iteratly adjust them
			-if classifier gets it right on data keep model, else if classifier's guess is wrong we modify the model
-Take Away, one way to think of learning is to training data to adjust the parameters of a model
http://playground.tensorflow.org 
^play around with neural networks
'''

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
'''
#different classifier that accomplishes the same thing as above one 
#Many different kinds of classifiers, high level they have same interface
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(x_train,y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print ("Accuracy of classifier predictions is ", accuracy_score(y_test, predictions))