import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA


# ######### GENERATING DATA ####################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations


# ########### PLOTTING THE DATA #################
# Original signals
colors = ['red', 'steelblue', 'orange']
plt.figure(figsize=[8,2])
for iSignal in range(3):
    plt.plot(np.arange(1,n_samples+1), S[:,iSignal], color=colors[iSignal])
plt.title('Original signals')
plt.show()

# mixed signals
mcolors = ['purple', 'springgreen','coral']
plt.figure(figsize=[8,2])
for iSignal in range(3):
    plt.plot(np.arange(1,n_samples+1), X[:,iSignal], color=mcolors[iSignal])
plt.title('Mixed signals')
plt.show()

# Correlation coefficients, original signals
print(np.corrcoef(S,rowvar=False))

# Correlation coefficients, mixed signals
print(np.corrcoef(X,rowvar=False))


# ######### ICA FIRST #########################
ica = FastICA(n_components=3)
X_ica = ica.fit_transform(X)  

# plotting ICs
pcolors = ['fuchsia','teal','gold']
plt.figure(figsize=[8,2])
for iSignal in range(3):
    plt.plot(np.arange(1,n_samples+1), X_ica[:,iSignal], color=pcolors[iSignal])
plt.title('Signals identified by ICA')
plt.show()



# ######### PCA NEXT #########################
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)  

# plotting PCs
plt.figure(figsize=[8,2])
for iSignal in range(3):
    plt.plot(np.arange(1,n_samples+1), X_pca[:,iSignal], color=pcolors[iSignal])
plt.title('Signals identified by PCA')
plt.show()

# correlation coefficients
print(np.corrcoef(X_pca,rowvar=False))




