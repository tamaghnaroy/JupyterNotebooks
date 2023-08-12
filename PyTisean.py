from pytisean import tiseano, tiseanio
import matplotlib.pyplot as plt
import numpy as np

# Generate 5000 iterates of the henon map
henon, msg = tiseano('henon', '-l5000')

# Plot and prettyfi
fig1, ax1 = plt.subplots(1, 1)
ax1.scatter(henon[:, 0], henon[:, 1], color='k', s=0.1)
ax1.set_title('The Henon map')
ax1.set_xlabel(r'$x$', fontsize=16)
ax1.set_ylabel(r'$y$', fontsize=16)
plt.show()


# Generate some data
N = 1000
t = np.linspace(0, N/10*np.pi, N)
x = np.sin(t) + 0.2*np.random.randn(N)
#  ... and compute the autocorrelation
acf, msg = tiseanio('corr', '-D', 50, data=x)

# Plot and prettyfi
bluish = '#2976bb' # https://xkcd.com/color/rgb/
fig2, ax2 = plt.subplots(2, 1)

ax2[0].set_title(r'Data and autodorrelation')
ax2[0].plot(t, x, color=bluish)
ax2[0].set_xlim(t[0], t[-1])
ax2[0].set_xlabel(r'$t$', fontsize=16)
ax2[0].set_ylabel(r'$x$', fontsize=16)

ax2[1].plot(acf[:, 0], acf[:, 1], color=bluish)
ax2[1].set_xlabel(r'Lag $k$', fontsize=16)
ax2[1].set_ylabel(r'ACF $\rho_x(k)$', fontsize=16)

plt.show()

n