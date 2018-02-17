# Step 1 Importing the required classes from scikit-learn
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

class irisDataSet:
# Step 2 Loading the iris dataset from scikit-learn
    iris = datasets.load_iris()


# Using the first two coloumns for classification of output variable
    X = iris.data[:, :2]
    y = iris.target
# Considering the value n_neighbors as 15
    n_neighbors = 15
    h = .02  # step size in the mesh

# Creating an ColourMap using below colour codes
    colourmap_light = ListedColormap(['#FA8072', '#32CD32', '#00FFFF'])
    colourmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Method KNNclassifier is used to classify the output variable among three categories based on first two coloumns
#Also, this method plots the Colourmap using matplotlib
    def KNNclassifier(self, n_neighbors,X,y,h,colourmap_light,colourmap_bold):
        for weights in ['uniform']:
            # we create an instance of Neighbours Classifier and fit the data.
            knnclf = neighbors.KNeighborsClassifier(n_neighbors)
            knnclf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = knnclf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plot.figure()
            plot.pcolormesh(xx, yy, Z, cmap=colourmap_light)

            # Plot also the training points
            plot.scatter(X[:, 0], X[:, 1], c=y, cmap=colourmap_bold,
                        edgecolor='k', s=20)
            plot.xlim(xx.min(), xx.max())
            plot.ylim(yy.min(), yy.max())
            plot.title("3-Class classification to classify species from iris data (k = %i, weights = '%s')"
                      % (n_neighbors, weights))
        #the show method is used to shoe the colour map for the knn classification of species from irir data
        plot.show()





def main():
    #Creating an object of Class irisDataSet
    d = irisDataSet()
    # using the above pbject of irisDataSet to fetch class variables and invoking method KNNclassifier
    d.KNNclassifier(d.n_neighbors,d.X,d.y,d.h,d.colourmap_light,d.colourmap_bold)


if __name__=="__main__":
    main()