'''
scikit includes sample data
in this example we will use the iris flower data
includes: 
-data from table of iris from wiki 
-meta data: names of features and names of different flowers
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

#print (iris.feature_names)
#print (iris.target_names)
#print (iris.data[0])
#print (iris.target_names[0])

test_idx = [0,50,100]

'''
Training hold majority of data to teach classifier -->training data
Purpose of removing so me data is to then be used against the now trained classifier 
and see how well classifier is at classifying the removed object -->testing data
	We know the answer before hand, thus a check to ensure it worked
'''
#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))

#visualization code
#following video
#edit for newer python:
#import pydot --> import pydotplus
#pydot.graph... --> pydotplus.graph...
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, 
	out_file = dot_data, 
	feature_names=iris.feature_names, 
	class_names=iris.target_names,
	filled=True, 
	rounded=True,
	impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")