'''
TensorFlow --> Good for Deep Learning like with Image Classifying
-Classifier we will be using is a Neural Network
'''

from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow.contrib import learn

#load dataset
iris = learn.datasets.load_dataset("iris")
x_train, x_test, y_train, y_test = model_selection.train_test_split(
	iris.data, 
	iris.target, 
	test_size=0.25, 
	random_state=42)

feature_cols = [tf.contrib.layers.real_valued_column("",dimension=4)]
#print(feature_cols)

'''
3 later deep neural network with 10, 20, 10 neurons respectively

3 layers come from length of hidden_units array only. 
Does not have anythong to do with "n_classes=3"

n_classes = 3 means there are 3 types of outputs (3 types of flowers)
'''
tensorflow_classifier = learn.DNNClassifier(
	hidden_units=[10, 20, 10],
	n_classes=3,
	feature_columns=feature_cols,
	model_dir="tmp/iris_model")

my_classifier = learn.SKCompat(tensorflow_classifier)

#fit and predict with scikit learn interface
my_classifier.fit(x_train, y_train, steps=200)
prediction = my_classifier.predict(x_test)
#print(prediction)

#accuracy 
score = metrics.accuracy_score(y_test, prediction["classes"])
print ("Accuracy is ", score)