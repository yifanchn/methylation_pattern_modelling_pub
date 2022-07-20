"""
Between sequence interactions considering independent sites.
"""
# modules from this repository:
from WithinSequence import withinsequence

# general imports:
import numpy as np
import math


# Parameter values
N = 4  # number of individuals
L = 100  # length of genomic region / sequence
tgen = 1000  # duration of one generation
tend = 5000  # stopping time

a = 1.25
s = 1.47e-3  # demethylation rate
y = 0.3  # strength of methylation vs demethylation. y<1 implies demethylated state is prefered.

k1 = s  # spontaneous demethylation
k2 = s * y  # spontaneous methylation
k3 = a * s  # demethylation due to neighbouring interaction
k4 = a * s * y  # methylation due to neighbouring interaction

# Define left and right neighbouring index
neighbour_l = np.zeros(L, int)
for l in range(L):
    if l == 0:
        neighbour_l[l] = L - 1
    else:
        neighbour_l[l] = l - 1

neighbour_r = np.zeros(L, int)
for l in range(L):
    if l == L - 1:
        neighbour_r[l] = 0
    else:
        neighbour_r[l] = l + 1

# X_cohort[n][i][l] stores the l-th site of the n-th individual in the i-th time point at which
# an event has happened.
X_cohort = []
for n in range(N):
    X_cohort.append(np.array([np.random.randint(2, size=L)]))

# t_cohort[n] stores the time evolution of the n-th individual.
t_cohort = []
for n in range(N):
    t_cohort.append([0])

m_cohort = []
for n in range(N):
    m_cohort.append([])

tgen_counter = 0  # tracks the number of generations that have passed

for i in range(math.floor(tend / tgen)):
    tgen_counter = (i + 1) * tgen
    for n in range(N):
        seq, has_tgen_reached, t, m = withinsequence(seq=[X_cohort[n][-1]], t=[t_cohort[n][-1]], L=L,
                                                     m=[m_cohort[n][-1]], a=1.25, s=s, tgen=tgen_counter)
        X_cohort[n] = np.append(X_cohort[n], seq[1:], axis=0)
        # We add another sequence of length L with only zeros. This will store the between
        # sequence interaction which happens at tgen_counter
        X_cohort[n] = np.append(X_cohort[n], [np.zeros((L,), dtype=int)], axis=0)
        t_cohort[n] = np.append(t_cohort[n], [t[1:], tgen_counter])
        m_cohort[n] = np.append(m_cohort[n], [m, 0])

    # The between sequence interaction after one generation looks as follows:
    for n in range(N):
        for l in range(L):
            frequency_methy = 0
            for m in range(N):
                frequency_methy = frequency_methy + X_cohort[m][-2][l] / L
        rand = np.random.uniform(0, 1)
        if rand <= frequency_methy:
            X_cohort[n][-1][l] = 1

# For the remaining time steps until tend, we will not have between sequence interactions anymore.
# Only within sequence interactions
for n in range(N):
    while t_cohort[n][-1] < tend:
        current_seq = X_cohort[n][-1]

        # Update number of methylated and unmethylated states denoted by counter1 and counter0, resp.
        counter0 = len(np.where(current_seq == 0)[0])
        counter1 = len(np.where(current_seq == 1)[0])

        # counter_xy checks the number of pairs that are in state xy, where y denotes the state of X_n and x
        # the state of the left neighbour
        counter01 = 0
        counter10 = 0
        for n in range(current_seq.shape[0]):
            if current_seq[n] == 0:
                if current_seq[neighbour_l[n]] == 1:
                    counter10 += 1
            else:
                if current_seq[neighbour_l[n]] == 0:
                    counter01 += 1

        r3 = counter01 * k3
        r4 = counter10 * k4
        r2 = counter0 * k2
        r1 = counter1 * k1

        # Update all possible rates
        possible_rates_total = [r3, r4, r2, r1]
        rate_sum = sum(possible_rates_total)

        # randomly choose next timepoint for an event to happen from an exponential distribution with mean
        # rate_sum
        tau = np.random.exponential(scale=(1 / rate_sum))
        t_cohort[n] = np.append(t_cohort[n], t_cohort[n][-1] + tau)

        # randomly choose a reaction from all (considering each site individually) possible reactions
        rand = rate_sum * np.random.uniform(0, 1)

        # spontaneous methylation
        if 0 <= rand <= r2:
            # look for all sites X_n = 0
            n = np.where(current_seq == 0)
            # randomly choose one of these from a uniform distribution
            chosen_n = np.random.randint(0, n[0].size)
            # change methylation state of this X_n
            current_seq[n[0][chosen_n]] = 1
        # spontaneous demethylation
        elif r2 < rand <= r2 + r1:
            n = np.where(current_seq == 1)
            chosen_n = np.random.randint(0, n[0].size)
            current_seq[n[0][chosen_n]] = 0
        # neighboring interaction demethylation for X_n = 1, X_{n-1} = 0
        elif r2 + r1 < rand <= r2 + r1 + r3:
            n = np.where(current_seq == 1)
            # check which X_n = 1 has a neighbour X_{n-1} = 0
            suitable_n = []
            for ind in n[0]:
                if current_seq[neighbour_l[ind]] == 0:
                    suitable_n.append(ind)
            chosen_n = np.random.randint(0, len(suitable_n))
            current_seq[suitable_n[chosen_n]] = 0
        # neighbouring interaction methylation for X_n = 0, X_{n-1} = 1
        else:
            n = np.where(current_seq == 0)
            # check which X_n = 0 has a neighbour X_{n-1} = 1
            suitable_n = []
            for ind in n[0]:
                if current_seq[neighbour_l[ind]] == 1:
                    suitable_n.append(ind)
            chosen_n = np.random.randint(0, len(suitable_n))
            current_seq[suitable_n[chosen_n]] = 1

        m_cohort[n] = np.append(m_cohort[n], len(np.where(current_seq == 1)[0]))
        X_cohort[n] = np.append(X_cohort[n], [current_seq], axis=0)
