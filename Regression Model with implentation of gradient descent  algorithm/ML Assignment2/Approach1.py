# Step 1 Importing the required classes from scikit-learn, numpy, pandas and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

class classification:

    # this function implements gradient descent algorithm to reduce the cost periodically
    def optimization_function(X, Y, beta, learning_rate, iterations):
        costs = [0] * iterations
        m = len(Y)
        for iteration in range(iterations):
            # defining the hypothesis as hβ(x)=βTx
            hypothesis = X.dot(beta)
            # Calculating loss at each iteration
            loss = hypothesis - Y
            # Calculating Gradient
            gradient = X.T.dot(loss) / m
            # Updating beta using Gradient Descent updation: βj:=βj−α(hβ(x)−y)xj)
            beta = beta - learning_rate * gradient
            # updateed Cost Value
            cost = classification.calculate_cost(X, Y, beta)
            costs[iteration] = cost
        return beta, costs

    # this function iteratively called optimization_function to calculate the cost at each updation of beta coefficients
    def calculate_cost(X, Y, beta):
         # the cost of the model is defined as J(β)=12m∑i=1m(hβ(x(i))−y(i))2
         # we need to reduce the value of cost J
         m = len(Y)
         J = np.sum((X.dot(beta) - Y) ** 2) / (2 * m)
         return J

    # this function calculates root_mean_square_error
    def root_mean_square_error(Y, predicted_output):
        root_mean_square_error = np.sqrt(sum((Y - predicted_output) ** 2) / len(Y))
        return root_mean_square_error

    def r2_score(Y, predicted_output):
        mean_y = np.mean(Y)
        ss_tot = sum((Y - mean_y) ** 2)
        ss_res = sum((Y - predicted_output) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def train_regression_model(self):
        # Step 2 Loading the iris dataset from scikit-learn
        iris = datasets.load_iris()
        #Storing the training data as input variables/Features in X and output variable(target) in y
        X = iris.data[:, :]
        y = iris.target
        #Spliting the iris data into two sets i.e. train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        x1 = X_train[:, 0]
        x2 = X_train[:, 1]
        x3 = X_train[:, 2]
        x4 = X_train[:, 3]
        m = len(x4)
        x0 = np.ones(m)
        X = np.array([x0, x1, x2, x3, x4]).T
        #Intialising the coefficients to beta vector
        beta = np.array([0, 0, 0, 0, 0])
        Y = np.array(y_train)
        #Intialising the learning rate for gradient descent algorithm
        learning_rate = 0.0005
        #reducing the cost prediocally using Gradient Descent algorithm.
        updatedbeta, costs = classification.optimization_function(X, Y, beta, learning_rate, 100000)
        cost_at_start = classification.calculate_cost(X, Y, beta)
        print("Cost before using Gradient Descent algorithm ")
        print(cost_at_start)
        # 100000 Iterations
        print("Final Coefficients")
        print(updatedbeta)

        # Final Cost of new beta
        print("Cost after using Gradient Descent algorithm")
        print(costs[-1])

        #storing the test data into coloumn vectors
        x5 = X_test[:, 0]
        x6 = X_test[:, 1]
        x7 = X_test[:, 2]
        x8 = X_test[:, 3]
        m = len(x5)
        x9 = np.ones(m)
        #combining the vectors to forn N dimensional array to satisfy matrix multiplication rules
        X5 = np.array([x9, x5, x6, x7, x8]).T

        #Predicting the output based on updated beta coefficients
        predicted_output = X5.dot(updatedbeta)
        print("root_mean_square_error")
        print(classification.root_mean_square_error(y_test, predicted_output))
        print("R square Score")
        print(classification.r2_score(y_test, predicted_output))
        light_color = ListedColormap(['#0000FF', '#62A9FF', '#99FF66'])
        plt.scatter(y_test, predicted_output, s=60, c=y_test, cmap=light_color, edgecolor='k')
        plt.show()

def main():
    #Creating an object of Class irisDataSet
    d = classification()
    # using the above pbject of irisDataSet to fetch class variables and invoking method KNNclassifier
    d.train_regression_model()


if __name__=="__main__":
    main()


