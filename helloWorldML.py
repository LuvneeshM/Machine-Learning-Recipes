from sklearn import tree

'''
features - used to describe data (i.e. weight, color)
label - identify the kind data (i.e. name)
'''
#in example: 0 = apple, 1 = orange
features = [[140,1],[130,1],[150,0],[170,0]] #input to classifier
labels = [0,0,1,1] #output from classifier

#decision tree - box of rules
clf = tree.DecisionTreeClassifier()
#think of fit as find patterns in data i.e. heavy weight = certain obj
clf = clf.fit(features,labels)
#input = features for new example
# will print what classifier thinks the object is
print (clf.predict([[160,0]]))