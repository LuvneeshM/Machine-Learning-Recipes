'''
What are good features
-Be able to identify differences between objects
-they take into account how distinguishable and unique
'''
import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height,lab_height], stacked=True, color=['r','b'])
plt.show()