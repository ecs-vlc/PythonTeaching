# Note - this code will only run realistically in jupyter notebook.

# 01-01

closed_list = []
method_list = []

def fib_closed(n):
    return ((1+np.sqrt(5))**n - (1-np.sqrt(5))**n) / ((2**n) * np.sqrt(5))

def fib_method(n):
    if n==0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_method(n-1) + fib_method(n-2)

n=20
fibs = np.empty((n,2))

for i in range(n):
    closed_list.append(fib_closed(i))
    method_list.append(fib_method(i))

print(closed_list)
print(method_list)

# 02-01

import numpy as np

def fisher_wright(P, s, mu, nu, Tmax):
	t = 0
	n = np.zeros((Tmax+1), dtype=np.int64)
	while n[t]<P and t<Tmax:
		# select
		p_s = (1+s)*n[t] / (P+s*n[t])
		# mutate
		p_sm = (1-nu)*p_s + mu*(1.-p_s)
		# sample
		t += 1
		n[t] = np.random.binomial(P, p_sm)
	return n[:t+1]


# 02-02

nt = fisher_wright(200, 0.1, 0.001, 0.001, int(10**3))
plt.plot(np.arange(len(nt)), nt, 'k--')
plt.xlabel("$t$")
plt.ylabel("$n$ mutants")

# 02-03

def fisher_wright_modified(P, s, mu, nu, Tmax, Nr):
    t = 0
    n = np.zeros((Tmax+1, Nr), dtype=np.int32)
    # once a certain proportion (say 10%) of the matrix contains total population P, we stop iterating.
    prop = ((Tmax+1)*Nr) * 0.1
    while np.sum(n==P) < prop and t < Tmax:
        p_s = (1+s)*n[t,:] / (P+s*n[t,:])
        # mutate
        p_sm = (1-nu)*p_s + mu*(1.-p_s)
        # sample
        t += 1
        n[t,:] = np.random.binomial(P, p_sm, size=(Nr,))
    return n[:t+1,:]

nt2 = fisher_wright_modified(200, 0.1, 0.001, 0.001, int(10**3), int(10**4))
ntm = nt2.mean(axis=1)
ntsd = nt2.std(axis=1)
t = np.arange(len(nt2))

# option 2. 
for i in range(nt2.shape[1]):
    plt.plot(t, nt2[:,i])
    
plt.plot(t, ntm, 'k-')
# option 1.
#plt.fill_between(t, ntm - 2*ntsd, ntm + 2*ntsd, color='r', alpha=.4)

plt.show()

# in depth cases:

#03-01-01

def gaussian_elimination(A, b):
    """
    Simple Gaussian Elimination algorithm with pivoting, using direct methods.
    Solves linear system Ax = b, assuming well-conditioned square matrix A.
    
    Parameters
    ----------
    A : matrix
        Square matrix, size n x n.
    b : vector
        RHS vector, size n (must conform with A).

    Returns
    -------
    x : vector
        unknown coefficients, size n
    """
    # Store size of system
    n = len(b)
    assert(np.all(A.shape == (n, n)))
    
    # Form augmented matrix
    aug = np.hstack((A, b.reshape(n,1)));
    
    # Loop over rows
    for i in range(n):
        # Find the row with largest magnitude, and then swap the rows.
        max_row = np.argmax(aug[i:, i])
        
        if (max_row): # Only swap rows if the maximum is not this row. \
                      # NOTE: the max_row is counted relative to i, so max_row = 0 => row i.
            tmp               = np.copy(aug[i, :])
            aug[i, :]         = np.copy(aug[i+max_row, :])
            aug[i+max_row, :] = np.copy(tmp)
        
        # Loop over rows below i
        for j in range(i+1, n):
            aug[j, :] -= (aug[j, i] / aug[i, i] * aug[i, :])
    
    # Return the separated, reduced, matrix and right hand side vector.
    return (aug[:,:-1], aug[:, -1])

A = np.array([[9, 3, 4], [4, 3, 4], [1, 1, 1]], dtype=np.float64)
b = np.array([7, 8, 3], dtype=np.float64)
Ast, bst = gaussian_elimination(A,b)
Ast, bst

# 03-01-02

def backward_substitution(Astar, bstar):
    """
    A* = reduced upper-triangular form
    b* = reduced vector
    
    Starts at bottom of matrix and works way up.
    """
    n,p = Astar.shape
    xstar = np.empty(p)
    
    xstar[-1] = bstar[-1] / Astar[-1,-1]
    for i in range(n-2, -1,-1):
        """
        print("i: %d" % i)
        print("x: %s" % xstar[i+1:])
        print("b: %s" % bstar[i])
        print("A: %s,%s" % (Astar[i,i+1:], Astar[i,i]))
        """
        xstar[i] = (bstar[i] - np.dot(xstar[i+1:], Astar[i,i+1:])) / Astar[i,i]
    return xstar

backward_substitution(Ast, bst)


# 04-01-01

def monte_carlo_integrate(f, dx, dy, N):
	area = (dx[1] - dx[0])*(dy[1] - dy[0])
	# generate random numbers in 2-d
	pairs = np.random.rand(N,2)
	# move pairs into domain [x,y]
	pairs[:,0] *= dx[1] - dx[0]
	pairs[:,0] += dx[0]
	pairs[:,1] *= dy[1] - dy[0]
	pairs[:,1] += dy[0]
	# x is in [:,0]
	integrand = f(pairs[:,0])
	# choose k where random numbers y fall below the integrand
	k = pairs[:,1] < integrand
	
	return (area * np.sum(k)) / N 

# 04-01-02

def f(x):
	return np.sin(1/(x*(2-x)))**2

I = monte_carlo_integrate(f, [0, 2], [0, 1], 10**5)
print(I)

# 04-01-03

def pi(x):
	return np.sqrt(4-x**2)

Nvals = 100*2**np.arange(0,15)
errs = np.zeros((15,))

for i, N in enumerate(Nvals):
	errs[i] = abs(monte_carlo_integrate(pi, [0, 2], [0, 2], N ) - np.pi)

# 04-01-04

plt.loglog(Nvals, errs, 'kx')

# 04-01-05

plt.loglog(Nvals, errs, 'kx')
m,b = np.polyfit(np.log(Nvals), np.log(errs), 1)
plt.loglog(Nvals, np.exp(m[1])*Nvals**fit[0], 'b--')

# 04-02-01

def least_squares(X, y):
	n, p = X.shape
	# add bias
	nX = np.column_stack((np.ones(n), X))
	# solve
	return np.dot(np.linalg.inv(np.dot(nX.T,nX)),np.dot(nX.T,y))

# 04-02-02

def ls_predict(X, w, bias_included=False):
    if bias_included:
        return np.dot(X,w)
    else:
        return np.dot(np.column_stack((np.ones(len(X)), X)), w)

np.random.seed(5458392)
X, y = make_regression()

# call method
w = least_squares(X, y)
# predict yp
yp = ls_predict(X, w)

plt.scatter(yp,y)
plt.plot([-.5, 3.], [-.5, 3.], 'k--')
plt.xlabel("predicted values")
plt.ylabel("actual values")
plt.title("Actual against predicted values")

print(w)

# 04-02-03

def ridge(X, y, lamda):
	n, p = X.shape
	# bias
	nX = np.column_stack((np.ones(n), X))
	# solve
	return np.dot(np.linalg.inv(np.dot(nX.T,nX) + lamda*np.eye(p+1)),np.dot(nX.T,y))

w = ridge(X, y, .1)
yp = ls_predict(X, w)
import matplotlib.pyplot as plt

print(w)

plt.scatter(yp, y)
plt.plot([-.5, 3.], [-.5, 3.], 'k--')
plt.xlabel("predicted values")
plt.ylabel("actual values")
plt.show()

# 04-02-04

def pearson(x, y):
    xm = x.mean()
    ym = y.mean()
    return np.sum((x - xm)*(y - ym)) / (np.sqrt(np.sum((x - xm)**2)) * np.sqrt(np.sum((y - ym)**2)))

p = pearson(yp, y)

plt.scatter(yp,y,label="r={:0.3f}".format(p))
plt.legend()
plt.show()

# 04-02-05

def gradient_descent(X, y, gamma = .001, n_iter=500):
	n, P = X.shape
	nX = np.column_stack(((np.ones(n,)), X))
	saved_w = np.empty((P+1, n_iter))
	w = np.random.rand(P+1)
	saved_w[:,0] = w
	for i in range(1,n_iter):
		dE = np.dot((2*nX.T),(np.dot(nX,w) - y))
		w -= gamma*dE
		saved_w[:,i] = w
	return saved_w, w

N_iter = 500

sw, w = gradient_descent(X, y, .001, N_iter)

print(X.shape, y.shape, sw.shape, w.shape)

t = np.arange(N_iter)
for i in range(len(w)):
	plt.plot(t,sw[i,:],'--',label=i)
plt.legend()
plt.show()

# 05-01-01

N = 500
dt = 1 / N
dW = np.sqrt(dt) * np.random.randn(N)
# set to 0 start
W = np.cumsum(dW) - dW[0]
t = np.linspace(0, 1, N)

plt.plot(t,W,'rx-')

# 05-01-02

M1 = 10
M2 = 10**5
dW = np.sqrt(dt) * np.random.randn(N,M1)
dW2 np.sqrt(dt) * np.random.randn(N,M2)
W1 = np.cumsum(dW, axis=0) - dW[0,:]
W2 =np.cumsum(dW2, axis=0) - dW2[0,:]
for i in range(M1):
	plt.plot(t, W[:,i])
plt.plot(t, np.mean(W1,axis=1),'k-')

plt.plot(t, np.mean(W2,axis=1), 'k-')

# 05-01-03

def euler_maruyama_step(X_n, dt, dW_n, lamda=2., mu=1.):
	fn = lamda * X_n
	gn = mu * X_n
	return X_n + fn*dt + gn*dW_n

# 05-01-04

def euler_maruyama(N, dt, dW, X_0):
	X = np.zeros(N)
	X[0] = X_0
	for n in range(N-1):
		X[n+1] = euler_maruyama_step(X[n], dt, dW[n], 2., 1.)
	return X

N = 100
dt = 1/N
dW = np.sqrt(dt)*np.random.randn(N)
X = euler_maruyama(N, dt, dW, 1.)

# 05-01-05

def f_exact(t, W, lamda, mu, X0):
	return X0 * np.exp((lamda = .5*mu**2)*t + mu*W)

t, dt = np.linspace(0, 1, N+1, retstep=True)
dW = np.sqrt(dt)*np.random.randn(N+1)
W = np.cumsum(dW) - dW[0]
X = euler_maruyama(len(t), dt, dW, 1.)
X_exact = f_exact(t, W, lamda, mu, X0)

plt.plot(t, X_exact, 'm-', 'expected')
plt.plot(t, X, 'r--*', label='realised')
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$X$")

# 05-02-01

def lennard_jones_potential(r):
	return 24.*(2.*(1/r)**14 - (1/r)**8)

# 05-02-02

def acceleration(x, L, Rc):
    a = np.zeros_like(x)
    N,P = x.shape
    for i in range(N):
        dx = x[i,:] - x
        s = np.abs(dx) > L/2
        dx[s] -= np.sign(dx[s])*L
        for j in range(i+1,N):
            rij = np.sqrt(np.dot(dx[j,:],dx[j,:]))
            if rij < Rc:
                phi_r = lennard_jones_potential(rij)
                a[i,:] += dx[j,:]*phi_r
                a[j,:] -= dx[j,:]*phi_r
    return a

# 05-02-03

def verlet(x, v, a, dt, L):
    x = x + dt * v + .5 * dt**2 * a
    # boundary check
    x[x < 0] += L
    x[x > L] -= L
    vstar = v + .5 * dt * a
    a = acceleration(x, L, 2.5)
    v = vstar + .5 * dt * a
    return x, v, a

# 05-02-04

from mpl_toolkits.mplot3d.axes3d import Axes3D
x = np.array([[4., 0., 0.],[4.+2.**(1/12),0.,0.]])
L = 10
Rc= 2.5
dt = 0.1
steps = 200
v = np.zeros_like(x)
a = acceleration(x, L, Rc)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

for i in range(steps):
    x, v, a = verlet(x, v, a, dt, L)
    ax.scatter(x[:,0], x[:,1], x[:,2])

# 05-02-05

def calc_temperature(v, L):
    E = np.zeros((len(v)))
    for particle in range(len(v)):
        E[particle] = 0.5*(L**2)*(np.linalg.norm(v[particle]))**2
    return np.sum(E)*(2./(3.*len(E)))

steps = 100
x = np.array([[4., 0., 0.],[4.+2.**(1/12),0.,0.]])
v = np.zeros_like(x)
a = acceleration(x, L, Rc)
dt = 0.005
L = 6.1984
T = np.zeros(steps)

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for i in range(steps):
    x, v, a = verlet(x, v, a, dt, L)
    T[i] = calc_temperature(v, L)
    ax2.scatter(x[:,0], x[:,1])

ax.plot(np.arange(0,steps*dt,dt), T, 'k-')
ax.set_xlabel("time t")
ax.set_ylabel("Temperature T")

# 05-02-6

from matplotlib import animation
from IPython.core.display import HTML

"""

def animate_graph(x, dt, steps, L, Rc):
    
    # mol dynamics
    N,D = x.shape
    # we want to save the variables.
    xnew = np.zeros((N,D,steps))
    xnew[:,:,0] = x
    v = np.zeros_like(x)
    a = acceleration(x, L, Rc)
    t = np.arange(0, dt*steps, dt)
    T = np.zeros(steps)
    
    # run algorithm
    for i in range(steps):
        x, v, a = verlet(x, v, a, dt, L)
        xnew[:,:,i] = x
        T[i] = calc_temperature(v, L)
    
    # calculate minmaxs
    x_min = np.min(xnew[:,0,:])
    y_min = np.min(xnew[:,1,:])
    z_min = np.min(xnew[:,2,:])
    x_max = np.max(xnew[:,0,:])
    y_max = np.max(xnew[:,1,:])
    z_max = np.max(xnew[:,2,:])
    
    # create figure
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    #ax2.set_zlim(0, 10)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$T$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax.set_xlim(0.0, dt*steps)
    ax.set_ylim(np.min(T) - np.min(T)/20, np.max(T) + np.max(T)/20)
    ax2.set_xlim(x_min - x_min/20, x_max + x_max/20)
    ax2.set_ylim(y_min - y_min/20, y_max + y_max/20)
    ax2.set_zlim(z_min - z_min/20, z_max + z_max/20)
    #ax2.set_zlabel("z")
    points_t, = ax.plot([], [], 'r-')
    points_3d, = ax2.plot([], [], [], linestyle="", marker="o")
    # close
    plt.close()
    
    print(T.shape)
    print(t.shape)
    print(xnew.shape)
    print(x.shape)
    
    def init():
        points_t.set_data([], [])
        points_3d.set_data([], [])
        return (points_t, points_3d)
    
    # with i being the index step
    def animate(i):
        # update points
        points_t.set_data(t[:i+1], T[:i+1])
        points_3d.set_data(xnew[:,0,i], xnew[:,1,i])
        points_3d.set_3d_properties(xnew[:,2,i])
        return (points_t, points_3d)
    
    return HTML(animation.FuncAnimation(fig, animate, init_func=init, 
                                        interval=40, frames=steps, blit=True).to_html5_video())


"""

x = np.loadtxt("input.dat", skiprows=1)

steps = 200
dt = 0.01
L = 6.1984
animate_graph(x, dt, steps, L, Rc)

