from pylab import *
from scipy import special

f = random.random()
print 1 - f
n = 1000
random1 = [1 if random.random() >= f else 0 for _ in range(n)]
alpha0 = beta0 = 1.
bprior0 = []

# computing prior distribution
for x in np.arange(0.01, 1.00, .01):
    bprior1 = 1. / special.beta(alpha0, beta0) * x ** (alpha0 - 1) * (1 - x) ** (beta0 - 1)
    bprior0.append(bprior1)

bprior0 = np.array(bprior0) / len(bprior0)

xaxis = np.linspace(0.01, .99, 99)

# updating posterior based on new data
for i in np.arange(0, 1000, 1):
    likelihood0 = []
    n, k = 1, random1[i]
    for p in np.arange(0.01, 1.00, .01):
        likelihood1 = special.binom(n, k) * p ** (k) * (1 - p) ** (n - k)
        likelihood0.append(likelihood1)
    likelihood0 = np.array(likelihood0)
    posterior = (likelihood0 * bprior0) / np.sum(likelihood0 * bprior0)
    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 999]:
        plt.xlabel('f')
        plt.ylabel('P(f)')
        plt.title('Posteriors for ' + str(i + 1) + ' tosses with f = ' + str(round(1 - f, 3)))
        plot(xaxis, posterior)
        show()
    bprior0 = posterior
