import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, norm, multivariate_normal
from scipy.special import logsumexp
from SER import SER_prior
from logistic_data_generator import logistic_generator

def log_sigmoid(beta, X):

    coeff =-X @ beta

    return -np.logaddexp(0,coeff)


def decode(encoded_particle):

    pass

class ProposalState:

    def __init__(self, omega, alphas, mus, vars):
        '''

        :param omega: 1 x N vector of transformed omega values
        :param alphas: L x M vector of variable probabilities for each variable
        :param mus: L x M vector of coefficient means for each variable
        :param vars: L x M vector of coefficient variances for each variable
        '''

        self.omega = omega
        self.alphas = np.array(alphas)
        self.mus = np.array(mus)
        self.vars = np.array(vars)

    def compute_deletion_probs(self):

        maxprobs = np.max(self.alphas, axis=1)
        delprobs = 1.0 - maxprobs

        total = np.sum(delprobs)

        if not np.isfinite(total):
            raise ValueError(
                f"Invalid deletion probabilities: "
                f"alphas contain non-finite values.\n{self.alphas}"
            )

        if total <= 0:
            # All components have probability 1 on one variable.
            # No information distinguishes which component to delete.
            return np.ones(len(delprobs)) / len(delprobs)

        return delprobs / total

    def sample_particle(self):
        g = np.zeros((len(self.alphas), len(self.alphas[0])))
        b = np.zeros((len(self.alphas), len(self.alphas[0])))

        for i in range(len(self.alphas)):

            j = np.random.choice([k for k in range(len(self.alphas[0]))], p = np.exp(self.alphas[i]))

            g[i,j] = 1
            b[i,j] = norm.rvs(self.mus[i,j], np.sqrt(self.vars[i,j]))

        return Particle(g,b)

class Particle:
    # A SuSiE particle comprising variable number, selections, coefficients of selected variables
    # and a particle weight - comments relate this to a dataset consisting of N results with M
    # measured independent variables (i.e the design matrix is N x M, the results a N x 1 vector)

    def __init__(self, gamma, beta):
        '''
        :param L: Number of components selected from independent variables
        :param gamma: L x M array of vectors, each row with a single entry equal to 1, the rest 0
        :param beta: L x M array of coefficients, each row with a single nonzero entry corresponding to the gammas
        :param state: State object containing current updated prior values
        '''


        self.gamma = gamma
        self.beta = beta


        self.L = gamma.shape[0]
        self.coefficient_array = np.sum(beta, axis = 0) # 1 x M vector of coefficients
        self.variables = len(self.coefficient_array)

        if gamma.shape[0] != self.L:
            raise ValueError(f"Number of variable selection vectors inconsistent with stated value of L")

        if beta.shape[0] != self.L:
            raise ValueError("Number of coefficient vectors inconsistent with stated value of L")

        if gamma.shape != beta.shape:
            raise ValueError(
                "Gamma and beta must have the same shape"
            )

    def particle_log_likelihood(self, X, y):
        '''

        :param X: N x M array of independent variable values
        :param y: N x 1 array of dependent variable values
        :return: log likelihood of current particle given the data
        '''


        if X.shape[1] != self.gamma.shape[1]:
            raise ValueError("Number of variables inconsistent with particle size")

        logp = log_sigmoid(self.coefficient_array, X)
        log_1mp = log_sigmoid(-self.coefficient_array, X)

        return np.sum(y*logp + (1-y)*log_1mp)

    def pad(self, max):
        '''

        :param max: integer
        :return: particle with zero padding up to max size
        '''
        pass

    def encode(self):
        '''

        :return: particle encoded for SMC sampler
        '''
        pass

    def compute_omega(self, X):
        '''

        :param X: N x M array of independent variable values
        :return: N x 1 array of omega values for the PG transform
        '''

        psi = X @ self.coefficient_array

        omega = np.empty_like(psi)

        small = np.abs(psi) < 1e-8
        omega[small] = 0.25

        omega[~small] = (
                0.5 / psi[~small]
                * np.tanh(psi[~small] / 2)
        )

        return omega


class InitProposal:

    #instantiates an initial particle with a specific log weight for a model with a specific number of variables

    def __init__(self, l_dist, pi, mus, vars):
        '''

        :param l_dist: N x 1 - discrete probability mass function for the initial number of contributing variables
        :param pick_dist: N x 1 Array - probability of each individual variable contributing
        :param mus: N x 1 Array of coefficient means
        :param vars: N x 1 Array of coefficient variances
        :param logw: Log weight of the initiated particle (defaults to zero i.e. weight 1 for MCMC)
        '''

        self.ldist = l_dist
        self.pi = pi
        self.mus = mus
        self.vars = vars
        self.m = len(self.pi)

        if (len(mus) != self.m) or (len(vars) != self.m):
            raise ValueError("Initiator object fails: alpha, mean and variance distribution arrays don't match")

        if not np.isclose(np.sum(self.ldist), 1):
            raise ValueError("Initiator object fails: L distribution does not sum to 1")

        if not np.isclose(np.sum(self.pi), 1):
            raise ValueError("Initiator object fails: alpha distribution does not sum to 1")


    def generate_initial_particle(self):

        l = np.random.choice(self.m, p=self.ldist) + 1

        g = np.zeros((l, self.m))
        b = np.zeros((l,self.m))

        for i in range(l):
            j = np.random.choice(self.m, p = self.pi)

            g[i,j] = 1
            b[i,j] = norm.rvs(self.mus[j], np.sqrt(self.vars[j]))

        return Particle(g, b)

    def compute_prior(self, particle):

        indices = [np.where(i == 1)[0][0] for i in particle.gamma]

        piprob = sum([np.log(self.pi[i]) for i in indices])
        betprob = sum([norm.logpdf(particle.beta[l, j],
                                   self.mus[j],
                                   np.sqrt(self.vars[j])
                                   )
                       for l, j in enumerate(indices)
        ])


        return piprob + betprob + np.log(self.ldist[len(particle.gamma) - 1])




class StateGenerator:

    def __init__(self, X, y, pi, mus, vars):
        '''

        :param X:
        :param y:
        :param pi:
        :param mu:
        :param var:
        '''

        self.X = X
        self.y = y

        # prior quantities
        self.pi = np.asarray(pi)
        self.mus = np.asarray(mus)
        self.vars = np.asarray(vars)

        self.m = X.shape[1]

        if len(self.pi) != self.m:
            raise ValueError("Inclusion probs inconsistent with data")

        if len(self.mus) != self.m:
            raise ValueError("Coefficient means inconsistent with data")

        if len(self.vars) != self.m:
            raise ValueError("Coefficient vars inconsistent with data")

    def compute_full_residual(self, particle):

        omega = particle.compute_omega(self.X)
        residual = (self.y - 0.5) - (omega * (self.X @ particle.coefficient_array))

        return residual

    def calculate_SER(self, residual, omega):

        bf = []
        mu = []
        var = []
        i = 0

        scales = []

        for x in self.X.T:
            wx = omega * x
            scale = x @ wx
            scales.append(scale)
            bvar = self.vars[i]

            bffact = -0.5 * np.log((1 + (bvar * scale)))
            bfexp = 0.5 * ((bvar * (x @ residual) ** 2 / (1 + (bvar * scale))))
            bf.append(bffact + bfexp)

            sig = 1 / ((1 / bvar) + scale)
            var.append(sig)
            mu.append(sig * (x @ residual))
            i += 1

        loga = np.log(self.pi) + np.asarray(bf)
        loga_norm = logsumexp(loga)

        log_alpha = loga - loga_norm

        return log_alpha, mu, var

    def calculate_state(self, particle):

        omega = particle.compute_omega(self.X)

        log_alphas = []
        mus = []
        vars = []

        for l in range(particle.L):
            other_beta = np.sum(
                np.delete(particle.beta, l, axis=0),
                axis=0
            )
            residual = (self.y - 0.5) - (omega*(self.X @ other_beta))

            log_a, m, v = self.calculate_SER(residual, omega)

            log_alphas.append(log_a)
            mus.append(m)
            vars.append(v)

        return ProposalState(omega, log_alphas, mus, vars)

class Proposal:

    # set of discrete and continuous proposals
    # proposal also includes an evaluator (which will give us the L-kernel)

    def __init__(self, proposal_generator, proposal_state, move_probs):
        '''

        :param proposal: Proposal_State type object
        :param move_probs: Probability of a forward, mid or backward move
        '''
        self.proposal_generator = proposal_generator
        self.proposal_state = proposal_state
        self.move_probs = move_probs

        if not np.isclose(np.sum(move_probs), 1):
            raise ValueError("Move probabilities don't sum to 1")

    def get_move_probs(self, L, state):
        """
        Return the actual move probabilities available from a particle
        with L components.
        """
        if L == 1:
            # Redistribute forbidden decrease probability equally
            # between static and increase.
            return np.array([
                0,
                self.move_probs[1] + self.move_probs[0],
                self.move_probs[2]
            ])
        elif L == len(state.mus[0]):
            return np.array([
                self.move_probs[0] + self.move_probs[2],
                self.move_probs[1],
                0
            ])

        return self.move_probs.copy()

    def forward_decrease(self, current):
        if current.L <= 1:
            raise ValueError("Cannot decrease below L=1")

        # Remove the last component, which is the reverse
        # of the increase operation that appends a component.
        new_particle = Particle(
            np.delete(current.gamma, -1, axis=0),
            np.delete(current.beta, -1, axis=0),
        )

        new_state = self.proposal_generator.calculate_state(
            new_particle
        )


        return new_particle, new_state

    def forward_decrease_eval(self, current, proposed, state):
        if current.L - proposed.L != 1:
            raise ValueError(
                "Incompatible particle sizes for decrease move"
            )



        # The proposed particle must be the current particle
        # with its final component removed.
        if not (
                np.array_equal(current.gamma[:-1], proposed.gamma)
                and np.array_equal(current.beta[:-1], proposed.beta)
        ):
            return -np.inf

        move_prob = self.get_move_probs(current.L, state)[0]

        if move_prob <= 0:
            return -np.inf

        return np.log(move_prob)

    def forward_static(self, particle):

        new_particle = self.proposal_state.sample_particle()
        new_state = self.proposal_generator.calculate_state(new_particle)

        return new_particle, new_state

    def forward_static_eval(self, current, proposed, state):

        if current.L != proposed.L:
            raise ValueError("Proposal is not static - lengths don't match")

        move_probs = self.get_move_probs(current.L, state)
        move_prob = move_probs[1]

        if move_prob <= 0:
            print("Static move probability is zero:", move_probs)
            return -np.inf

        logprob = np.log(move_prob)

        for i in range(current.L):
            comp = np.where(proposed.gamma[i] == 1)[0][0]
            b = proposed.beta[i, comp]

            log_alpha = state.alphas[i, comp]
            mu = state.mus[i, comp]
            var = state.vars[i, comp]

            if log_alpha == -np.inf:
                print(f"Bad alpha: row={i}, comp={comp}, alpha={log_alpha}")
                return -np.inf

            if not np.isfinite(var) or var <= 0:
                print(f"Invalid variance: row={i}, comp={comp}, var={var}")
                return -np.inf

            log_beta = norm.logpdf(
                b, loc=mu, scale=np.sqrt(var)
            )

            if not np.isfinite(log_beta):
                print(
                    f"Non-finite coefficient density: row={i}, comp={comp}, "
                    f"b={b}, mu={mu}, var={var}, logpdf={log_beta}"
                )

            logprob += log_alpha + log_beta

            component_logprobs = []

            for i in range(current.L):
                comp = np.where(proposed.gamma[i] == 1)[0][0]

                log_alpha = state.alphas[i, comp]

                log_beta = norm.logpdf(
                    proposed.beta[i, comp],
                    loc=state.mus[i, comp],
                    scale=np.sqrt(state.vars[i, comp])
                )

                component_logprobs.append((log_alpha, log_beta))

        return logprob


    '''
    def forward_static_eval(self, current, proposed, state):

        if current.L != proposed.L:
            raise ValueError("Proposal is not static - lengths don't match")

        logprob = np.log(self.get_move_probs(current.L, state)[1])

        for i in range(current.L):
            comp = np.where(proposed.gamma[i] == 1)[0][0]
            b = proposed.beta[i, comp]

            logprob += np.log(state.alphas[i, comp])

            logprob += norm.logpdf(
                b,
                loc=state.mus[i, comp],
                scale=np.sqrt(state.vars[i, comp])
            )


        return logprob
    '''

    def forward_increase(self, current):

        #select new variable
        omega = current.compute_omega(self.proposal_generator.X)
        residuals = self.proposal_generator.compute_full_residual(current)

        a, m, s = self.proposal_generator.calculate_SER(residuals, omega)

        new_g = np.zeros_like(current.beta[0])
        new_b = np.zeros_like(current.beta[0])

        new_comp = np.random.choice(len(a), p = np.exp(a))

        new_g[new_comp] = 1
        new_b[new_comp] = norm.rvs(m[new_comp], np.sqrt(s[new_comp]))

        new_gammas = np.vstack((current.gamma, new_g))
        new_betas = np.vstack((current.beta, new_b))

        new_particle = Particle(new_gammas, new_betas)

        new_state = self.proposal_generator.calculate_state(new_particle)

        return new_particle, new_state

    def forward_increase_eval(self, current, proposed, state):

        if proposed.L - current.L != 1:
            raise ValueError("Increase proposal should be one longer than current")

        # Find which component was added by identifying the row whose
        # removal from `proposed` reproduces `current`.
        matching_indices = []

        for i in range(proposed.L):
            test_gamma = np.delete(proposed.gamma, i, axis=0)
            test_beta = np.delete(proposed.beta, i, axis=0)

            if (
                    np.array_equal(test_gamma, current.gamma)
                    and np.array_equal(test_beta, current.beta)
            ):
                matching_indices.append(i)

        if not matching_indices:
            raise ValueError(
                "Could not identify the component added in the increase proposal"
            )

        # In the usual case there is one matching index. If two components
        # are completely identical, either one gives the same proposed
        # particle, so their proposal probabilities should be summed.
        omega = current.compute_omega(self.proposal_generator.X)
        residuals = self.proposal_generator.compute_full_residual(current)

        a, m, s = self.proposal_generator.calculate_SER(residuals, omega)

        added_log_probs = []

        for i in matching_indices:
            g = proposed.gamma[i]
            comp = np.where(g == 1)[0][0]
            b = proposed.beta[i, comp]

            coeff_prob = norm.logpdf(
                b,
                loc=m[comp],
                scale=np.sqrt(s[comp])
            )

            added_log_probs.append(
                coeff_prob + a[comp]
            )

        # If multiple identical rows can be removed to give the same
        # proposed state, sum their probabilities.
        component_log_prob = logsumexp(added_log_probs)

        move_prob = self.get_move_probs(current.L, state)[2]

        return component_log_prob + np.log(move_prob)

    def forward(self, current, state):

        move_probs = self.get_move_probs(current.L, state)

        r = np.random.choice(3, p=move_probs)
        if r == 0:

            particle, state = self.forward_decrease(current)

        elif r == 1:
            particle, state = self.forward_static(current)
        else:
            particle, state = self.forward_increase(current)

        return particle, state

    def eval(self, current, proposed, state):

        d = len(proposed.gamma) - len(current.gamma)

        if np.abs(d) > 1:
            raise ValueError(f"Proposal particle length not compatible with current particle length - difference is {d}")

        if d == 1:
            return self.forward_increase_eval(current, proposed, state)
        elif d == 0:
            return self.forward_static_eval(current, proposed, state)
        else:
            return self.forward_decrease_eval(current, proposed, state)



