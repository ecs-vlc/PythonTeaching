
# 01-01

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(3, 2)

x = np.linspace(-5*np.pi, 5*np.pi, 1000)
y = [np.sin(x), np.cos(x), np.sin(3*x), 2*np.sin(x), np.tan(x-1)]
c = ['r', 'g', 'b', 'k', 'orange']
markers = ['x', 'o', '*', '^', 'd']

for i,eq in enumerate(['$\sin x$', '$\cos x$', '$\sin 3x$', '$2 \sin x$']):
    ax = fig.add_subplot(gs[int(i/2), i%2])
    ax.plot(x, y[i], c=c[i], marker=markers[i])
    ax.set_ylabel(eq)
    ax.set_xlabel("$x$")

ax = fig.add_subplot(gs[2,:])
ax.plot(x, y[-1], c=c[-1], marker=markers[-1])
ax.set_ylabel(r'$\tan x-1$')
ax.set_xlabel("$x$")
ax.set_ylim(-5,5)
fig.tight_layout()

# 02-01

titanic.assign(lived=titanic.Survived.replace({0:'Died',1:'Survived'})).groupby("lived").Age.plot.kde(xlim=(0,100), legend=True)

# 03-01

rain_yr = long_rain.resample("Y").median()

rain_yr["1920":"1960"].plot()
plt.xlabel("Time")
plt.ylabel("Rainfall")
plt.title("Rainfall between 1920-1960")
plt.show()

# 03-02

fig,ax=plt.subplots(2, figsize=(12,8))

d_rain = long_rain - long_rain.shift()

ax[0].plot(long_rain)
ax[1].plot(d_rain, label=r"$\Delta$ rain", alpha=.5)
ax[1].plot(d_rain.rolling(30).mean(), label="rolling mean")
ax[1].plot(d_rain.rolling(30).std(), label="rolling std")
ax[1].legend()
plt.show()

# 03-03

fig,ax=plt.subplots(2, figsize=(12,8))

rain_ewm = rain_yr.ewm(halflife=6).mean()
rain_ewm_diff = rain_yr - rain_ewm

ax[0].plot(rain_yr, label="input")
ax[0].plot(rain_ewm, label="ewm 12")
ax[1].plot(rain_ewm_diff, label="input_diff")
ax[1].plot(rain_ewm_diff.rolling(7, center=True).mean(), label="rolling mean 7")
ax[1].plot(rain_ewm_diff.rolling(7, center=True).std(), label="rolling std 7")

[a.legend() for a in ax]

plt.show()

# 03-04

decomp = seasonal_decompose(d_rain["2000":].dropna())

fig,ax=plt.subplots(4, figsize=(8,10))
ax[0].plot(d_rain, label="input", color='k')
ax[1].plot(decomp.trend, label="trend", color='r')
ax[2].plot(decomp.seasonal, label="seasonal", color='g')
ax[3].plot(decomp.resid, label="residual", color='b')
for a in ax:
    a.legend()
plt.show()




