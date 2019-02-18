import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
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
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
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




# producing a toy data set: three clusters, touching
X, y = make_blobs(n_samples=150, centers=3,
                  random_state=24, cluster_std=3.0)


# plotting the toy data
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plt.show()


# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=50, 
                                                    random_state=0)


# SVM model fitting
sv = SVC(kernel='linear', C=1.0)
sv.fit(X_train,y_train)


# plotting the decision boundaries from the SVM
plt.figure(figsize=[6,6])
ax = plt.subplot(111)
plot_contours(ax, sv, X[:, 0], X[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
            s=50, cmap=plt.cm.coolwarm)
plt.title('Decision boundaries with the training data')
plt.show()



# SVM classifier
y_pred = sv.predict(X_test)   # predicted class
y_margin = sv.decision_function(X_test)  # margins to boundaries


# plotting the margins for different classes 
targetColors = ['magenta', 'blue', 'green']
obsVec = np.arange(1,len(y_margin)+1)
plt.figure(figsize=[8,4])
for iClass in range(3):
    plt.plot(obsVec,y_margin[:,iClass], 
             ls='-', c=targetColors[iClass])
    plt.plot(obsVec[y_pred==iClass],2.5*np.ones_like(obsVec[y_pred==iClass]),
             marker = '^', ls='none', c=targetColors[iClass])
plt.ylim([-0.6, 2.6])
plt.xlabel('Observations')
plt.ylabel('Margin to the boundary')
plt.title('Margin distance and classification outcome')
plt.show()


# plotting the boundaries and the testing data
plt.figure(figsize=[6,6])
ax = plt.subplot(111)
plot_contours(ax, sv, X[:, 0], X[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries with the testing data')
plt.show()


# Confusion matrix
print(confusion_matrix(y_test,y_pred))

# classification report
print(classification_report(y_test, y_pred))

