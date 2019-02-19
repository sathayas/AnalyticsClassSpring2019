import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
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


# Loading the iris data
iris = datasets.load_iris()
X = iris.data[:,[0,3]]  # sepal length and petal width only
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=2018)

# SVM on training data (Linear)
svLin = SVC(kernel='linear',C=1.0)
svLin.fit(X_train,y_train)

# Predicted classes (Linear)
y_pred_Lin = svLin.predict(X_test)

# Confusion matrix (Linear)
print(confusion_matrix(y_test,y_pred_Lin))

# classification report (Linear)
print(classification_report(y_test, y_pred_Lin,
                            target_names=target_names))



# SVM on training data (RBF)
svRBF = SVC(kernel='rbf',C=1.0)
svRBF.fit(X_train,y_train)

# Predicted classes (RBF)
y_pred_RBF = svRBF.predict(X_test)

# Confusion matrix (RBF)
print(confusion_matrix(y_test,y_pred_RBF))

# classification report (RBF)
print(classification_report(y_test, y_pred_RBF,
                            target_names=target_names))



# SVM on training data (Polynomial)
svPoly = SVC(kernel='poly',C=1.0)
svPoly.fit(X_train,y_train)

# Predicted classes (Polynomial)
y_pred_Poly = svPoly.predict(X_test)

# Confusion matrix (Polynomial)
print(confusion_matrix(y_test,y_pred_Poly))

# classification report (Polynomial)
print(classification_report(y_test, y_pred_Poly,
                            target_names=target_names))



# plotting the boundaries and the testing data
plt.figure(figsize=[9,4])
ax = plt.subplot(131)
plot_contours(ax, svLin, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries (Linear)\nwith testing data')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

ax = plt.subplot(132)
plot_contours(ax, svRBF, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries (RBF)\nwith testing data')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

ax = plt.subplot(133)
plot_contours(ax, svPoly, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries (Poly)\nwith testing data')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

plt.subplots_adjust(wspace=0.4, bottom=0.15, left=0.075, right=0.975)
plt.show()


