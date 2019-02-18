import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# producing a toy data set: two clusters, separated
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=10, cluster_std=2.5)


# plotting the toy data
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plt.show()


# plotting possible boundaries
xMin = -2.5
xMax = 9.0
xfit = np.linspace(xMin, xMax)
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
for m, b in [(0.875, -6.5), (-0.2, -1), (1.8, -10)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(xMin, xMax)
plt.show()


# plotting possible boundaries with margins
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
for m, b, d in [(0.875, -6.5, 1.9), (-0.2, -1, 0.7), (1.8, -10, 1.05)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, 'k-')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='m', alpha=0.4)
plt.xlim(xMin, xMax)
plt.show()


# SVM
sv = SVC(kernel='linear', C=10000)
sv.fit(X,y)


# plotting the descision boundary with SVM, with margins
# as well as support vectors
plt.figure(figsize=[6,6])
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)

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

plt.show()


# coordinate for the first SV
print(sv.support_vectors_[0,:])

# coordinate for the second SV
print(sv.support_vectors_[1,:])


### SVM Classifier ###

# split the toy data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=10, 
                                                    random_state=0)

# SVM fitting
sv_train = SVC(kernel='linear', C=10000)
sv_train.fit(X_train,y_train)

# SVM classifier
y_pred = sv_train.predict(X_test)

# plotting the training data and classification outcome
plt.figure(figsize=[6,6])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='+',
            label='Training data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='^',
            label='Classification results')
plt.legend()
plt.show()
