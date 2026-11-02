
import copy
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

import IBSS

def particle_signature(p):
    return (
        p.L,
        tuple(p.gamma.flatten()),
        tuple(p.beta.flatten()),
    )

class RJSMC:
    def __init__(
        self,
        initiator,
        generator,
        proposal,
        ess_threshold=0.5,
        ess_target=0.7,
        rejuvenation_steps=1,
    ):
        if not 0 < ess_threshold <= ess_target < 1:
            raise ValueError(
                "Require 0 < ess_threshold <= ess_target < 1."
            )

        self.initiator = initiator
        self.generator = generator
        self.proposal = proposal

        # Fraction of N at which resampling is triggered.
        self.ess_threshold = ess_threshold

        # Target fraction of N when choosing the next temperature.
        self.ess_target = ess_target

        # Number of MH rejuvenation moves per particle per stage.
        self.rejuvenation_steps = rejuvenation_steps

    def initiate_particles(self, N):
        particles = [
            self.initiator.generate_initial_particle()
            for _ in range(N)
        ]

        log_weights = np.full(N, -np.log(N))
        return particles, log_weights

    def log_likelihood(self, particle):
        return particle.particle_log_likelihood(
            self.generator.X,
            self.generator.y,
        )

    def log_prior(self, particle):
        return self.initiator.compute_prior(particle)

    @staticmethod
    def normalised_weights(log_weights):
        return np.exp(log_weights - logsumexp(log_weights))

    def effective_sample_size(self, log_weights):
        w = self.normalised_weights(log_weights)

        #print("ESS:", 1.0 / np.sum(w ** 2))
        #print("Sum weights:", np.sum(w))
        #print("Nonzero weights:", np.count_nonzero(w))
        #print("Unique weights:", len(np.unique(np.round(w, 12))))

        return 1.0 / np.sum(w**2)

    def systematic_resample(self, particles, log_weights):

        print(
            "Unique particles before:",
            len({particle_signature(p) for p in particles})
        )

        N = len(particles)
        weights = self.normalised_weights(log_weights)

        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0

        positions = (
            np.random.uniform() + np.arange(N)
        ) / N

        indices = np.searchsorted(cumulative, positions)

        # Copies prevent accidental mutation of shared particle objects.
        new_particles = [
            copy.deepcopy(particles[i]) for i in indices
        ]

        new_log_weights = np.full(N, -np.log(N))

        print(
            "Unique particles after:",
            len({particle_signature(p) for p in new_particles})
        )

        return new_particles, new_log_weights

    def choose_next_temperature(
        self,
        beta,
        log_weights,
        log_likelihoods,
    ):
        """
        Find the largest beta_new in [beta, 1] for which
        the predicted ESS is at least ess_target * N,
        unless beta_new=1 already satisfies that condition.
        """
        N = len(log_weights)
        target_ess = self.ess_target * N

        def predicted_ess(beta_new):
            delta = beta_new - beta

            candidate = (
                log_weights + delta * log_likelihoods
            )

            candidate -= logsumexp(candidate)

            return self.effective_sample_size(candidate)

        # Can finish annealing in one step if possible.
        if predicted_ess(1.0) >= target_ess:
            return 1.0

        low = beta
        high = 1.0

        # Bisection to find the next temperature.
        for _ in range(60):
            mid = (low + high) / 2.0

            if predicted_ess(mid) >= target_ess:
                low = mid
            else:
                high = mid

        beta_new = low

        # Avoid getting stuck due to floating-point precision.
        if beta_new <= beta + 1e-10:
            return min(1.0, beta + 1e-6)

        return beta_new

    def reweight(
        self,
        particles,
        log_weights,
        beta_old,
        beta_new,
        log_likelihoods=None,
    ):
        if log_likelihoods is None:
            log_likelihoods = np.array([
                self.log_likelihood(p) for p in particles
            ])

        delta_beta = beta_new - beta_old

        new_log_weights = (
            log_weights + delta_beta * log_likelihoods
        )

        # Keep log weights normalised.
        new_log_weights -= logsumexp(new_log_weights)

        return new_log_weights, log_likelihoods

    def rejuvenate(self, particles, beta):
        """
        Apply MH moves targeting:
            log prior + beta * log likelihood.

        SMC weights do not change during rejuvenation.
        """
        new_particles = []
        accepted = 0
        attempted = 0

        for original_particle in particles:
            particle = copy.deepcopy(original_particle)

            for _ in range(self.rejuvenation_steps):
                current_state = self.generator.calculate_state(
                    particle
                )

                self.proposal.proposal_state = current_state

                proposed, proposed_state = self.proposal.forward(
                    particle, current_state
                )

                # Proposal densities, assuming eval(a, b, state)
                # returns log q(b | a).
                log_q_forward = self.proposal.eval(
                    particle, proposed, current_state
                )

                log_q_reverse = self.proposal.eval(
                    proposed, particle, proposed_state
                )


                #print(f"L = {particle.L} -> L = {proposed.L}")
                #print(f"log q forward: {log_q_forward}")
                #print(f"log q reverse: {log_q_reverse}")

                log_prior_current = self.log_prior(particle)
                log_prior_proposed = self.log_prior(proposed)

                loglik_current = self.log_likelihood(particle)
                loglik_proposed = self.log_likelihood(proposed)

                prior_term = log_prior_proposed - log_prior_current
                likelihood_term = beta * (loglik_proposed - loglik_current)
                proposal_term = log_q_reverse - log_q_forward

                log_alpha = prior_term + likelihood_term + proposal_term

                log_u = np.log(max(np.random.uniform(), np.finfo(float).tiny))
                accepted_move = log_u < min(0.0, log_alpha)

                if accepted_move:
                    particle = proposed
                    accepted += 1

                attempted += 1

                if np.log(np.random.uniform()) < min(
                    0.0, log_alpha
                ):
                    particle = proposed
                    accepted += 1

            new_particles.append(particle)

        acceptance_rate = (
            accepted / attempted if attempted else 0.0
        )

        return new_particles, acceptance_rate

    def run(self, N, T=100, burn=0):
        """
        Adaptive annealed SMC.

        N: number of particles.
        T: maximum number of annealing stages.
        burn: number of initial stages to omit from history.
              This does not discard particles from the algorithm.
        """
        if N < 2:
            raise ValueError("N must be at least 2.")
        if T < 1:
            raise ValueError("T must be at least 1.")
        if burn < 0:
            raise ValueError("burn cannot be negative.")

        particles, log_weights = self.initiate_particles(N)
        beta = 0.0
        beta_new = 0.0

        history = {
            "particles": [],
            "states": [],
            "log_weights": [],
            "weights": [],
            "beta": [],
            "ess": [],
            "log_likelihood": [],
            "particle_sizes": [],
            "resampled": [],
            "acceptance_rate": [],
        }

        for stage in range(T):
            print(f"Step {stage}")
            # Likelihood values are calculated at the current particles.
            log_likelihoods = np.array([
                self.log_likelihood(p) for p in particles
            ])

            # If current ESS is already below the annealing target,
            # resample first so that a meaningful temperature increment
            # can be selected.
            resampled = False

            if self.effective_sample_size(log_weights) < (
                self.ess_target * N
            ):
                particles, log_weights = self.systematic_resample(
                    particles, log_weights
                )
                resampled = True

                log_likelihoods = np.array([
                    self.log_likelihood(p) for p in particles
                ])

            print(
                f"beta={beta:.6f}, "
                f"ESS={self.effective_sample_size(log_weights):.1f}, "
                f"loglik min={np.min(log_likelihoods):.2f}, "
                f"loglik max={np.max(log_likelihoods):.2f}, "
                f"loglik sd={np.std(log_likelihoods):.2f}"
            )

            beta_new += 1/T

            '''
            beta_new = self.choose_next_temperature(
                beta,
                log_weights,
                log_likelihoods,
            )
            '''

            # Incremental importance weighting.
            log_weights, log_likelihoods = self.reweight(
                particles,
                log_weights,
                beta,
                beta_new,
                log_likelihoods,
            )

            beta += 1/T

            ess = self.effective_sample_size(log_weights)

            # Resample when ESS falls below the resampling threshold.
            if ess < self.ess_threshold * N:
                particles, log_weights = self.systematic_resample(
                    particles, log_weights
                )
                resampled = True

            # MH moves target the newly tempered posterior.
            particles, acceptance_rate = self.rejuvenate(
                particles, beta_new
            )

            beta = beta_new

            print(
                f"Stage {stage + 1}/{T} | "
                f"beta={beta:.6f} | "
                f"ESS={self.effective_sample_size(log_weights):.1f}/"
                f"{N} | "
                f"resampled={resampled} | "
                f"acceptance={acceptance_rate:.3f} |"
                f"average size={np.average([i.L for i in particles])} "
            )

            if stage >= burn:
                history["particles"].append(
                    copy.deepcopy(particles)
                )
                history["states"].append([
                    self.generator.calculate_state(p)
                    for p in particles
                ])
                history["log_weights"].append(
                    log_weights.copy()
                )
                history["weights"].append(
                    self.normalised_weights(log_weights)
                )
                history["beta"].append(beta)
                history["ess"].append(
                    self.effective_sample_size(log_weights)
                )
                history["log_likelihood"].append([
                    self.log_likelihood(p) for p in particles
                ])
                history["particle_sizes"].append([
                    p.L for p in particles
                ])
                history["resampled"].append(resampled)
                history["acceptance_rate"].append(
                    acceptance_rate
                )

            if beta >= 1.0 - 1e-10:
                beta = 1.0

        if beta < 1.0 - 1e-10:
            warnings.warn(
                f"Annealing stopped at beta={beta:.6f}, "
                f"before reaching the full posterior (beta=1). "
                "Increase T or investigate very small temperature steps."
            )

        history["final_beta"] = beta
        history["final_particles"] = particles
        history["final_log_weights"] = log_weights.copy()
        history["final_weights"] = self.normalised_weights(
            log_weights
        )

        return history

from logistic_data_generator import logistic_generator

p = 20
n = 1000

beta = np.zeros(p)

beta[3] = 1
beta[7] = 1
beta[12] = 1
beta[15] = 1

linkvars = {4:3, 8:7, 13:12}

test_LG = logistic_generator(p, beta, "Binary", linkvars=linkvars)

X,y = test_LG.generate_random_data(n)

y = y.T

initiator = IBSS.InitProposal([1/20]*20,
                              [1/20]*20,
                              [0]*20,
                              [1]*20
                              )

generator = IBSS.StateGenerator(X,
                                y,
                                [1/50]*20,
                                [0]*20,
                                [1]*20
                        )

proposal = IBSS.Proposal(generator, None, [1/3, 1/3, 1/3])

rjsmc = RJSMC(
    initiator,
    generator,
    proposal,
    ess_threshold=0.5,
    ess_target=0.7,
    rejuvenation_steps=1,
)

history = rjsmc.run(
    N=20,
    T=1000,
    burn=0,
)

def plot_alpha_pmfs(history, generator, max_L=None):

    particles = history["final_particles"]
    weights = history["final_weights"]

    # Calculate proposal states for the final particles
    states = [
        generator.calculate_state(p)
        for p in particles
    ]

    L_values = sorted(set(p.L for p in particles))

    if max_L is not None:
        L_values = [L for L in L_values if L <= max_L]

    n_variables = len(particles[0].gamma[0])

    fig, axes = plt.subplots(
        len(L_values), 1,
        figsize=(12, 3 * len(L_values)),
        sharex=True
    )

    if len(L_values) == 1:
        axes = [axes]

    for ax, L in zip(axes, L_values):

        # Select particles with this value of L
        indices = [
            i for i, p in enumerate(particles)
            if p.L == L
        ]

        # Normalise weights conditional on L
        conditional_weights = weights[indices]
        conditional_weights = (
            conditional_weights / conditional_weights.sum()
        )

        alpha_pmf = np.zeros(n_variables)

        for idx, weight in zip(indices, conditional_weights):

            # Stored alphas are log probabilities
            alphas = np.exp(states[idx].alphas)

            # Average over the L components of this particle
            particle_alpha = np.mean(alphas, axis=0)

            alpha_pmf += weight * particle_alpha

        ax.bar(np.arange(n_variables), alpha_pmf)

        ax.set_title(f"L = {L} (n = {len(indices)} particles)")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Variable index")
    axes[-1].set_xticks(np.arange(n_variables))

    plt.tight_layout()
    plt.show()

particles = history["final_particles"]
weights = history["final_weights"]

for p in range(len(particles)):
    particle = particles[p]

    selected = [
        np.where(row == 1)[0][0]
        for row in particle.gamma
    ]

    unique_selected = set(selected)

    print(
        f"L={particle.L}, "
        f"distinct variables={len(unique_selected)}, "
        f"selections={selected}"
    )

max_variables = 20

posterior_K = np.zeros(max_variables + 1)

for particle, weight in zip(particles, weights):
    selected = [
        np.where(row == 1)[0][0]
        for row in particle.gamma
    ]

    K = len(set(selected))
    posterior_K[K] += weight

for K, probability in enumerate(posterior_K):
    if probability > 0:
        print(f"{K} distinct variables: {probability:.3f}")

import numpy as np
import matplotlib.pyplot as plt
import os


import numpy as np
import matplotlib.pyplot as plt
import os


def save_gamma_component_plots(history, output_folder="gamma_component_plots",
                               n_plots=20):

    particles = history["final_particles"]
    weights = history["final_weights"]

    n_variables = particles[0].gamma.shape[1]

    os.makedirs(output_folder, exist_ok=True)

    for l in range(n_plots):

        # Find all particles containing component l
        indices = [
            i for i, p in enumerate(particles)
            if p.L > l
        ]

        if len(indices) == 0:
            print(f"Component {l + 1}: no particles contain this component.")
            continue

        # Normalise weights conditional on the component existing
        component_weights = weights[indices].copy()
        component_weights /= component_weights.sum()

        # Weighted selection frequency for each variable
        variable_probabilities = np.zeros(n_variables)

        for idx, weight in zip(indices, component_weights):

            selected_variable = np.argmax(
                particles[idx].gamma[l]
            )

            variable_probabilities[selected_variable] += weight

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.bar(
            np.arange(n_variables),
            variable_probabilities
        )

        ax.set_title(
            f"Component {l + 1}: weighted variable selection"
        )
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Selection probability")
        ax.set_xticks(np.arange(n_variables))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filename = os.path.join(
            output_folder,
            f"component_{l + 1:02d}.png"
        )

        plt.savefig(filename, dpi=150)
        plt.close(fig)

        print(
            f"Saved {filename} "
            f"(based on {len(indices)} particles)"
        )



def plot_beta_pmfs(history):

    particles = history["final_particles"]
    weights = history["final_weights"]

    L_values = sorted(set(p.L for p in particles))
    n_variables = particles[0].beta.shape[1]

    fig, axes = plt.subplots(
        len(L_values), 1,
        figsize=(12, 3 * len(L_values)),
        sharex=True
    )

    if len(L_values) == 1:
        axes = [axes]

    for ax, L in zip(axes, L_values):

        indices = [
            i for i, p in enumerate(particles)
            if p.L == L
        ]

        # Normalise weights conditional on L
        conditional_weights = weights[indices]
        conditional_weights /= conditional_weights.sum()

        weighted_beta = np.zeros(n_variables)

        for idx, weight in zip(indices, conditional_weights):

            # Sum coefficients across components
            beta_total = np.sum(particles[idx].beta, axis=0)

            weighted_beta += weight * beta_total

        ax.bar(
            np.arange(n_variables),
            weighted_beta
        )

        ax.set_title(f"L = {L} (n = {len(indices)} particles)")
        ax.set_ylabel("Weighted mean coefficient")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Variable index")
    axes[-1].set_xticks(np.arange(n_variables))

    plt.tight_layout()
    plt.show()

save_gamma_component_plots(history)