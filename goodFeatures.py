'''
What are good features
-Be able to identify differences between objects
-they take into account how distinguishable and unique
-in ml you need multiple features --better for features to be independent, avoid the redundant features
-types of features: think of what you need to know,unique traits, that would help

Recap: Ideal features are
-Informative
-Independent
-Simple
'''
import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

#as shown by chart, while height does help identify which kind of dog well for the outer cases
#is shows to be difficult to only use height for the middle section/area
#would need another feature such as how fast god runs, average weight, average hair length
#to further narrow down the kind of dog for the classifier to decide
plt.hist([grey_height,lab_height], stacked=True, color=['r','b'])
plt.show()