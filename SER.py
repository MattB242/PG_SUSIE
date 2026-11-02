import numpy as np
import rdata
import matplotlib.pyplot as plt
from scipy.stats import multinomial, norm
from scipy.special import logsumexp
from logistic_data_generator import logistic_generator

class SER_prior:

    def __init__(self, pi, beta_means, beta_vars, noise_var):

        self.d = len(beta_vars)
        self.pi = pi
        self.beta_means = beta_means
        self.beta_vars = beta_vars
        self.noise_var = noise_var

    def sample_prior(self):
        gamma_vec = multinomial.rvs(1, self.pi)
        gamma = np.argmax(gamma_vec)
        beta = norm.rvs(self.beta_means[gamma], np.sqrt(self.beta_vars[gamma]))

        return gamma, beta

    def SER(self, X, y):
        """

        :param self:
        :param X: Input features
        :param y: Output data
        :return:
        """
        bf = []
        mu = []
        var = []
        i = 0
        for x in X.T:
            scale = x.T @ x
            print(f"scale range: {min(scale), max(scale)}")
            bvar = self.beta_vars[i]
            bet_est = (x.T @ y)/scale
            var_est = scale

            #print(f"s={var_est}, bhat = {bet_est}")

            bffact = np.log(np.sqrt(var_est/(bvar + var_est)))
            bfexp = (
                    (bet_est ** 2) * bvar
                    / (2 * var_est * (bvar + var_est))
            )

            #print(f"fact={bffact}, exp = {bfexp}")

            bf.append(bffact + bfexp)
            sig = 1/((1/var_est) + (1/self.beta_vars[i]))
            var.append(sig)
            mu.append((bet_est*sig)/var_est)
            i +=1

        loga = np.log(self.pi) + np.asarray(bf)
        loga_norm = logsumexp(loga)

        alpha = np.exp(loga-loga_norm)



        return alpha, mu, var

    def SER_PG(self, X, y, omega):
        """

        :param self:
        :param X: Input features
        :param y: Output data
        :return:
        """
        bf = []
        mu = []
        var = []
        i = 0

        scales = []
        for x in X.T:
            wx = omega * x
            scale = x @ wx
            scales.append(scale)
            bvar = self.beta_vars[i]


            bffact = -0.5*np.log((1+(bvar*scale)))
            bfexp = 0.5*((bvar*(x@y)**2/(1+(bvar*scale))))
            bf.append(bffact + bfexp)

            sig = 1/((1/bvar) + scale)
            var.append(sig)
            mu.append(sig*(x @ y))
            i += 1

        print(f"max mean size {np.max(np.abs(mu))}")
        print(f"max var size {np.max(np.abs(sig))}")
        print(f"scale range {max(scales)-min(scales)}")
        loga = np.log(self.pi) + np.asarray(bf)
        loga_norm = logsumexp(loga)

        alpha = np.exp(loga - loga_norm)

        return alpha, mu, var

'''

p = 20
n = 10000

beta = np.zeros(p)

beta[3] = 1
beta[7] = 1
beta[12] = 1

linkvars = {4:3, 8:7, 13:12}

#test_LG = logistic_generator(p, beta, "Binary", linkvars)

#X,y = test_LG.generate_random_data(n)
#print(X.shape)



test_SER = SER_prior(1/p*np.ones(p), np.zeros(p), np.ones(p), 1)

b = np.zeros(p)
L = 50
n, p = X.shape

alpha = [np.ones(p) / p for _ in range(L)]
mu = [np.zeros(p) for _ in range(L)]
var = [np.ones(p) for _ in range(L)]

gamma_counts = [np.zeros(p) for _ in range(L)]

gamma_inits = [np.random.choice(p, p=a) for a in alpha]


beta_all = np.zeros((L,p))

# fitted mean contribution
mu_hat = np.zeros(p)
b = np.zeros(n) + 1e-08   # fitted signal in observation space: X @ beta
atot = np.array(alpha)
rplot = []

for it in range(30000):
    beta_total = np.sum(beta_all, axis = 0)
    psi = X @ beta_total

    omega = np.empty_like(psi)

    small = np.abs(psi) < 1e-8
    omega[small] = 0.25

    omega[~small] = (
            0.5 / psi[~small]
            * np.tanh(psi[~small] / 2)
    )

    for l in range(L):

        beta_total -= beta_all[l]

        r = (y-0.5) - (omega*(X @ beta_total))
        print(np.dot(r,r))

        a, m, s = test_SER.SER_PG(X, r, omega)

        gamma = np.random.choice(p, p=a)
        if it > 10000:
            gamma_counts[l][gamma] += 1

        beta = norm.rvs(m[gamma], np.sqrt(s[gamma]))
        beta_all[l, :] = 0
        beta_all[l, gamma] = beta

        beta_total += beta_all[l]

        alpha[l] = a
        mu[l] = m
        var[l]  = s

        #diagnostics
        ind = np.argsort(alpha[l])[-5:]
        al = [alpha[l][j] for j in ind]
        print(l, ind, al)

    if it > 10000:
        atot += alpha


    rfull = (y - 0.5) - (omega * (X @ beta_total))
    rplot.append(np.sum(rfull))

output_path = "C:/Users/mattb242/Desktop/Projects/joint_signal_detection/MULTIDRUG/pythonProject/results/PG_in_SuSiE/"


coeffs = np.array(beta_total)
incprobs = np.array(atot)/20000

print(f"Final beta inclusion average: {coeffs}")
print(f"Final alpha average: {incprobs}")

varind = [i+1 for i in range(20)]

v = 0
for i in incprobs:
    v+=1
    plt.bar(varind, i, label = f"Variable {v}")

plt.ylabel("p(inclusion)")
plt.xlabel("Dependent Variable")
plt.xticks(varind)
plt.legend()
plt.savefig(output_path + "synthetic_data_inclusion")
plt.cla()

plt.bar(varind, coeffs)
plt.ylabel("Mean (beta)")
plt.xlabel("Dependent Variable")
plt.xticks(varind)
plt.savefig(output_path + "synthetic_data_coefficients")
plt.cla()


'''



