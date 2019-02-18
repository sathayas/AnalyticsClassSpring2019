import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
from sklearn.svm import SVC

def plot_svm_margin(X,y,sv):
    '''
    Input:
         X:  2D data matrix
         y:  Target vector
         sv: Support vector machine results
    Returns:
         None
    Produces:
         A scatter plot of the data X, with colors defined by y.
         SVM boundary, as well as the margin is plotted. Also
         support vectors are indicated.

         You need to run plt.show() after everything is done.
    '''
    # scatter plot first
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.coolwarm)
    # suppor for the meshgrid
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = sv.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(sv.support_vectors_[:, 0], sv.support_vectors_[:, 1], s=200,
               linewidth=1, edgecolors='k', facecolors='none')


# Creating a toy data with circles
X, y = make_circles(100, factor=.1, noise=.1, random_state=88)

# plotting the data
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.coolwarm)
plt.show()

# calculating the radius
r = np.sum(X**2, axis=1)**0.5

# plotting the data, y-axis replaced with the radius
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], r, c=y, s=50, cmap=plt.cm.coolwarm)
plt.show()


# SVM
R = np.vstack([X[:, 0], r]).T
sv = SVC(kernel='linear', C=10000)
sv.fit(R,y)


# plotting the margin on the SVM of the transformed data
plt.figure(figsize=[6,6])
plot_svm_margin(R,y,sv)
plt.show()


# SVM with RBF kernel
svRBF = SVC(kernel='rbf', C=10000)
svRBF.fit(X,y)

# plotting the margin on the SVM with the RBF kernel
plt.figure(figsize=[6,6])
plot_svm_margin(X,y,svRBF)
plt.show()
