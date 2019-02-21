import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
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


# Loading the iris data
iris = datasets.load_iris()
X = iris.data[:,[0,3]]  # sepal length and petal width only
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=2018)




# k nearest neighbor classifier objects
kNN5uni = KNeighborsClassifier(5, weights='uniform')
kNN5uni.fit(X_train,y_train)

kNN5dist = KNeighborsClassifier(5, weights='distance')
kNN5dist.fit(X_train,y_train)

kNN20uni = KNeighborsClassifier(20, weights='uniform')
kNN20uni.fit(X_train,y_train)

kNN20dist = KNeighborsClassifier(20, weights='distance')
kNN20dist.fit(X_train,y_train)



# Predicted classes
y_pred_5uni = kNN5uni.predict(X_test)
y_pred_5dist = kNN5dist.predict(X_test)
y_pred_20uni = kNN20uni.predict(X_test)
y_pred_20dist = kNN20dist.predict(X_test)


# plotting the boundaries and the testing data
plt.figure(figsize=[9,9])
ax = plt.subplot(221)
plot_contours(ax, kNN5uni, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries\n(k=5, uniform)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

ax = plt.subplot(222)
plot_contours(ax, kNN20uni, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries\n(k=20, uniform)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

ax = plt.subplot(223)
plot_contours(ax, kNN5dist, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries\n(k=5, distance)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

ax = plt.subplot(224)
plot_contours(ax, kNN20dist, X_train[:, 0], X_train[:, 1],
                  cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:,0], X_test[:,1],
            marker = '^', c=y_test,
            cmap=plt.cm.coolwarm)
plt.title('Decision boundaries\n(k=20, distance)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

plt.subplots_adjust(hspace=0.3, wspace=0.275, 
                    bottom=0.05, top=0.95, left=0.10, right=0.975)
plt.show()


# classifier performance
print('K=5, uniform\n', confusion_matrix(y_test,y_pred_5uni))
print('\nK=5, distance\n', confusion_matrix(y_test,y_pred_5dist))
print('\nK=20, uniform\n', confusion_matrix(y_test,y_pred_20uni))
print('\nK=20, distance\n', confusion_matrix(y_test,y_pred_20dist))

print(classification_report(y_test, y_pred_20uni,
                            target_names=target_names))
