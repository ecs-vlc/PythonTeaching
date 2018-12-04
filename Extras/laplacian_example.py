
import numpy as np
import matplotlib.pyplot as plt

def create_random_adjacency(size=500, blocks=100, sets=[0, 200, 320, 390], missing=.9999, set_density=.6):
    # create matrix
    x = np.random.rand(size,size)
    np.putmask(x, x<missing, 0.)
    np.putmask(x, x>=missing, 2.)
    x = np.tril(x, k=-1)
    x += x.T
    # create sets
    y = np.random.rand(blocks, blocks)
    np.putmask(y, y<set_density, 0.)
    np.putmask(y, y>=set_density, 2.)
    y = np.tril(y, k=-1)
    y += y.T
    # add sets to x
    for s in sets:
        x[s:s+blocks,s:s+blocks] = y
    
    xnew = np.empty_like(x)
    # shuffle and permutate each axis using same seed
    np.random.seed(123456789)
    np.take(x, np.random.permutation(x.shape[0]), axis=0, out=xnew)
    np.random.seed(123456789)
    np.take(xnew, np.random.permutation(x.shape[1]), axis=1, out=xnew)
    # degrees is calculated from sum of each column, then convert to matrix
    D = np.diagflat(np.sum(x, axis=0))
    # return degree, adjac
    return D, xnew, x


Dx, Ax, Ax_old = create_random_adjacency(missing=.99)
print(np.allclose(Ax, Ax.T, atol=1e-8), np.allclose(Dx, Dx.T, atol=1e-8))
fig,ax=plt.subplots(ncols=2)
ax[0].imshow(Ax, cmap="Greys")
ax[1].imshow(Ax_old, cmap="Greys")

# create laplacian
Lx = Dx - Ax

# calculate eigs
lamda, v = np.linalg.eig(Lx)

plt.plot(np.sort(v[:,1]))
plt.show()

sorted_v = np.argsort(v[:,1])
mag_sort = np.argsort(np.linalg.norm(Ax,axis=1))

# plot sorted 2nd eigenvector
fig,ax=plt.subplots(ncols=2)
ax[0].imshow(Ax[sorted_v][:,sorted_v], cmap="Greys")
ax[1].imshow(Ax[new_sort][:,new_sort], cmap="Greys")
plt.show()

# normalize?
Dx_norm = np.diagflat(1. / np.diag(np.sqrt(Dx)))
Lx_norm = np.dot(Dx_norm, np.dot(Lx, Dx_norm))

lamda2, v2 = np.linalg.eig(Lx_norm)
sorted_v2 = np.argsort(v2[:,1])

# significantly better!
fig,ax=plt.subplots(ncols=2, figsize=(8,5))
ax[0].imshow(Ax[sorted_v2][:,sorted_v2], cmap="Greys")
ax[1].plot(np.sort(v2[:,1]))
plt.show()