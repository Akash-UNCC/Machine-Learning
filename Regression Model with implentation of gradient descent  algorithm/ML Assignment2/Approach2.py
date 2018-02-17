import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
import numpy.polynomial.polynomial as poly


class classification:

     # created a constructor to initalse
     def __init__(self,X,y):
         self.X = X
         self.y = y

     def train_regression_model(self):
         iris = datasets.load_iris()
         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=4)
         light_color = ListedColormap(['#0000FF','#62A9FF','#99FF66'])
         X_train = X_train[:,3]
         X_test  = X_test[:,3]
         coefs = poly.polyfit(X_train, y_train, 4)
         ffit = poly.polyval(X_test, coefs)
         print(ffit)
         plt.scatter(y_test, ffit,s=60,c=y_test,cmap=light_color,edgecolor='k')
         plt.show()


def main():
    #Creating an object of Class irisDataSet
    iris = datasets.load_iris()

    d = classification(iris.data[:,:],iris.target)
    #
    d.train_regression_model()


if __name__=="__main__":
    main()
