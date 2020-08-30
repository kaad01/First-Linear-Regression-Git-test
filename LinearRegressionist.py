import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("xTest.txt");                            #read Dataset
"""
this textfile has to contain two columns labled x and y. With the data you want to use.
"""
x = data[["x"]]
y = data[["y"]]

#uncomment this in your first run to create a pickle file, which will contain a linearregressionmodel
"""
#best accuracy gets saved
best = 0
for i in range(50):
    #splitting the data randomly in training and testing sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()                                #Creating Enviroment

    linear.fit(x_train, y_train)                                            #Training (creating our graph)
    acc = linear.score(x_test, y_test)                                      #Testing an calculating the distance of our prediction and the actual value

    #compare acc
    if acc > best:
        best = acc
        # Save our most accurate linear model with pickle
        with open("LinearRegressionist.pickle", "wb") as f:
            pickle.dump(linear, f)

print(best)
"""
linear = pickle.load(open("LinearRegressionist.pickle", "rb"))                     #load our model

#plot Data
style.use("ggplot")
plt.xlabel("x")
plt.ylabel("y")

plt.scatter(data["x"], data["y"])
y_pred = linear.predict(x)
plt.plot(x, y_pred, color = "blue")                                                 #plot our prediction
plt.show()

#predictions
while (True):
    x_input = input("Which x do you want to predict?\n")
    if x_input == "quit":
        break
    x_input = int(x_input)
    y_input = linear.predict(np.array(x_input).reshape(1,-1))
    print(str(x_input) + " is " + str(y_input[0][0]))

