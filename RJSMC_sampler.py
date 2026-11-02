import numpy as np
import IBSS
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def sort_particle_components(particle):
    indices = np.argmax(particle.gamma, axis=1)
    order = np.argsort(indices, kind='stable')

    particle.gamma = particle.gamma[order]
    particle.beta = particle.beta[order]

    particle.coefficient_array = np.sum(particle.beta, axis=0)

    return particle

def RJMCMC(initiator : IBSS.InitProposal,
           generator : IBSS.StateGenerator,
           T: int,
           burn: int,
           ess_threshold = 0.5,
           fixed_move_prob = [1/3, 1/3, 1/3]):

    if burn > T:
        raise ValueError("Burn in longer than overall time!")

    particle = initiator.generate_initial_particle()
    state = generator.calculate_state(particle)

    proposal = IBSS.Proposal(generator, state, fixed_move_prob)

    t = 0
    ac = 0
    particle_sample = []

    while t < T+1:
        print(f"Current particle size: {particle.L}")
        print(f"Step {t}")
        new_particle, new_state = proposal.forward(particle)

        eval_ratio = proposal.eval(particle, new_particle, state) - proposal.eval(new_particle, particle, new_state)
        likelihood_ratio = new_particle.particle_log_likelihood(generator.X, generator.y) - particle.particle_log_likelihood(generator.X, generator.y)
        prior_ratio = initiator.compute_prior(new_particle) - initiator.compute_prior(particle)

        u = np.random.uniform()

        if u < np.exp(eval_ratio + likelihood_ratio + prior_ratio):
            particle, state = new_particle, new_state
            print("move accepted!")
            ac += 1

        proposal = IBSS.Proposal(generator, state, fixed_move_prob)


        if t > burn:
            particle_sample.append(particle)

        t += 1

    print(f"acceptance rate: {ac/t}")
    return np.array(particle_sample)


class RJSMC:

    def __init__(self,
                 initiator : IBSS.InitProposal,
                 generator : IBSS.StateGenerator,
                 proposal : IBSS.Proposal,
                 ess_threshold = 0.5
                 ):

        self.initiator = initiator
        self.generator = generator
        self.proposal = proposal
        self.resampled = True
        self.ess_threshold = ess_threshold


    def initiate_particles(self, N):

        particles = []
        weights = []
        for i in range(N):
            particle = sort_particle_components(self.initiator.generate_initial_particle())

            particles.append(particle)
            weights.append(-np.log(N))

        return particles, weights




    def propagate(self, particles, weights):

        current_states = [self.generator.calculate_state(i) for i in particles]

        new_particles = []
        new_weights = []

        for i in range(len(particles)):

            self.proposal.proposal_state = current_states[i]

            new_particle, new_state = self.proposal.forward(particles[i], current_states[i])

            new_particle = sort_particle_components(new_particle)

            new_particles.append(new_particle)

            eval_ratio =  self.proposal.eval(new_particles[-1], particles[i], new_state) - self.proposal.eval(particles[i], new_particles[-1], current_states[i])

            likelihood_ratio = (particles[i].particle_log_likelihood(self.generator.X, self.generator.y) -new_particles[-1].particle_log_likelihood(self.generator.X, self.generator.y))

            prior_ratio = self.initiator.compute_prior(new_particles[-1]) - self.initiator.compute_prior(particles[i])

            new_weights.append(eval_ratio + likelihood_ratio + prior_ratio + weights[i])

        return new_particles, new_weights

    def normalised_weights(self, weights):

        log_norm = logsumexp(weights)

        return np.exp(weights - log_norm)

    def effective_sample_size(self, weights):

        w = self.normalised_weights(weights)

        return 1.0 / np.sum(w ** 2)

    def reweight(self, particles, weights, beta_old, beta_new):
        """Importance-weight update for the next tempered target."""

        delta_beta = beta_new - beta_old

        new_weights = np.asarray(weights, dtype=float).copy()

        for i, particle in enumerate(particles):
            loglik = particle.particle_log_likelihood(
                self.generator.X,
                self.generator.y
            )
            new_weights[i] += delta_beta * loglik

        return new_weights

    def rejuvenate(self, particles, beta):
        """Apply MH moves targeting prior * likelihood**beta."""

        new_particles = []
        accepted = []

        for particle in particles:
            state = self.generator.calculate_state(particle)
            self.proposal.proposal_state = state

            proposed, proposed_state = self.proposal.forward(
                particle, state
            )

            log_q_forward = self.proposal.eval(
                particle, proposed, state
            )
            log_q_reverse = self.proposal.eval(
                proposed, particle, proposed_state
            )

            ll_current = particle.particle_log_likelihood(
                self.generator.X, self.generator.y
            )
            ll_proposed = proposed.particle_log_likelihood(
                self.generator.X, self.generator.y
            )

            prior_current = self.initiator.compute_prior(particle)
            prior_proposed = self.initiator.compute_prior(proposed)

            log_alpha = (
                    beta * (ll_proposed - ll_current)
                    + prior_proposed - prior_current
                    + log_q_reverse - log_q_forward
            )

            accept = (
                    np.log(np.random.uniform())
                    < min(0.0, log_alpha)
            )

            if accept:
                new_particles.append(proposed)
            else:
                new_particles.append(particle)

            accepted.append(accept)

        return new_particles, np.mean(accepted)


    def systematic_resample(self, particles, weights):

        w = self.normalised_weights(weights)

        s = np.cumsum(w)

        positions = (np.arange(len(weights)) + np.random.uniform()) / len(weights)

        indices = np.searchsorted(
            s,
            positions
        )

        particles = [
            particles[i]
            for i in indices
        ]

        weights = [-np.log(len(weights))]*len(weights)

        return particles, weights

    def one_step(self, particles, weights):

        new_particles, new_weights = self.propagate(particles, weights)

        #new_weights = self.reweight(particles, weights, b_old, b_new)

        if self.effective_sample_size(new_weights) < self.ess_threshold:

            self.resampled = True
            new_particles, new_weights = self.systematic_resample(particles, new_weights)

        else:
            self.resampled = False

        return new_particles, new_weights

    def run(self, N : int, T : int, burn = 0):

        if burn >= T:
            raise ValueError(f"Run length {T} should be greater than burn-in {burn}")

        particles, weights = self.initiate_particles(N)

        diagnostic_dict = {"particles": [],
                           "states" : [],
                           "ess": [],
                           "log_likelihood": [],
                           "particle_sizes": [],
                           "resampled" :[]
                           }

        #linear_temp = np.linspace(0,1,T)

        for _ in range(T):

            print(f"step {_}")
            print(f"particle lengths are: {[i.L for i in particles]}")
            print(f"mean particle length: {np.mean([i.L for i in particles])}")

            particles, weights = self.one_step(particles, weights)

            #print(f"First betas are {[i.beta[0] for i in particles]}")
            seconds = []
            for i in particles:
                if i.L == 1:
                    seconds.append("na")
                else:
                    seconds.append(np.where(i.gamma[1] == 1)[0][0])
            print(f"Second variables are {seconds}")

            if _ > burn:
                diagnostic_dict["particles"].append(particles)
                diagnostic_dict["states"].append([self.generator.calculate_state(i) for i in particles])
                diagnostic_dict["ess"].append(self.effective_sample_size(weights))
                diagnostic_dict["log_likelihood"].append([i.particle_log_likelihood(self.generator.X, self.generator.y)
                                                      for i in particles])
                diagnostic_dict["particle_sizes"].append([i.L for i in particles])
                diagnostic_dict["resampled"].append(self.resampled)

        return diagnostic_dict


from logistic_data_generator import logistic_generator

p = 50
n = 1000

beta = np.zeros(p)

beta[3] = 1
beta[7] = 1
beta[12] = 1
beta[15] = 1

#linkvars = {4:3, 8:7, 13:12}

test_LG = logistic_generator(p, beta, "Binary")

X,y = test_LG.generate_random_data(n)

y = y.T

initiator = IBSS.InitProposal([1/50]*50,
                              [1/50]*50,
                              [0]*50,
                              [1]*50
                              )

generator = IBSS.StateGenerator(X,
                                y,
                                [1/50]*50,
                                [0]*50,
                                [1]*50
                        )

proposal = IBSS.Proposal(generator, None, [1/3, 1/3, 1/3])

RJSMC = RJSMC(initiator, generator, proposal)

SMC_stats = RJSMC.run(20, 1000, 100)

loglik = np.mean(SMC_stats["log_likelihood"], axis = 1)

plt.plot([i for i in range(len(loglik))], loglik)
plt.show()
