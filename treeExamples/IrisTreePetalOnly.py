import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = np.min(x,axis=0) - 1, np.max(x,axis=0) + 1
    y_min, y_max = np.min(y,axis=0) - 1, np.max(y,axis=0) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, x, y, h=.02, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    params: dictionary of params to pass to contourf, optional
    """
    xx, yy = make_meshgrid(x, y, h)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Loading data
iris = load_iris()
X = iris.data[:,2:] # focusing on petal features only 
y = iris.target
feature_names = iris.feature_names[2:]
target_names = iris.target_names

# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=2018)


# decision tree classifier
dt = DecisionTreeClassifier(criterion='entropy', 
                            min_samples_leaf = 3,
                            max_depth = 4,
                            random_state=0)
dt.fit(X_train,y_train)


# visualizing the decision tree
dot_data = export_graphviz(dt, feature_names=feature_names,
                           class_names=target_names, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None)
graph = graphviz.Source(dot_data)

### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph



# classification on the testing data set
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=target_names))


# plotting the boundaries and the testing data
plt.figure(figsize=[6,6])
ax = plt.subplot(111)
plot_contours(ax, dt, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries\nwith testing data')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

