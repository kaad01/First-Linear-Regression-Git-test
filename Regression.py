import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student/student-mat.csv", sep=";")                  #read Dataset
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]    #format Dataset

predict = "G3"

X = np.array(data.drop(predict))                                   #Everything without Predict
Y = np.array(data[predict])                                             #Predicts
#splitting the data randomly in training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""                                                                     #We already have the a very accurate linear model
#best accuracy gets saved
best = 0
for i in range(50):

    #splitting the data randomly in training and testing sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()                                #Creating Enviroment

    linear.fit(x_train, y_train)                                            #Training (creating our graph)
    acc = linear.score(x_test, y_test)                                      #Testing an calculating the distance of our prediction and the actual value

    #compare acc
    if acc > best:
        best = acc
        # Save our most accurate linear model with pickle
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print(best)
"""

linear = pickle.load(open("studentmodel.pickle", "rb"))                     #load our model


#Plotting the data
x_axe = "G1"
style.use("ggplot")
pyplot.scatter(data[x_axe], data["G3"])
pyplot.xlabel(x_axe)
pyplot.ylabel("Final Grade")
pyplot.show()