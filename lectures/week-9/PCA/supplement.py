Xpca = pca.transform(X)

plt.scatter(Xpca[:,0], Xpca[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


pca = PCA(n_components=1)
pca.fit(X)
Xpca = pca.transform(X)
X_new = pca.inverse_transform(Xpca)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.6)
plt.axis('equal');

## After Standard Scaling
## Note that in general using PCA with StantardScaler is a good idea
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

plt.scatter(X[:,0], X[:,1])
plt.show()


rng = np.random.RandomState(2)
X = rng.multivariate_normal(mean = [0,0], cov = [[100, 0.1], [0.1, 1]], size = 200)
plt.scatter(X[:, 0], X[:, 1] )
plt.axis('scaled')
plt.draw()

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

plt.scatter(X[:,0], X[:,1])
plt.axis('scaled')
plt.show()

X_inv  = pca.inverse_transform(Xpca)

plt.scatter(X_inv[:,0], X_inv[:,1])

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2, color = 'r',
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1])
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')