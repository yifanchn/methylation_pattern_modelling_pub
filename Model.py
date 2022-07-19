"""
Stochastic simulation of our (new modified) models using Gillespie's algorithm.
one_neighbour describes the model considering only the left nearest neighbour.
both_neighbours describes the model considering both nearest neighbours.
Each of the functions stores the time series, the methylation sequence at each time step as well as
the methylation level (number of methylated sites) at each time step.
"""

import numpy as np


def one_neighbour(init_seq=None, tend=50, a=1.25, s=1.47e-3, y=0.3, L=100):
    """
    Runs the Gillespie algorithm accounting only for the left nearest neighbour
    :param init_seq: initial sequence. Has the form np.array([[0, 1, ..., 1]])
    :param tend: stopping time
    :param a: parameter value that indicates the strength of neighbouring interaction, a>1
    :param s: demethylation rate
    :param y: strength of methylation vs. demethylation, y<1: demethylated state preferred state
    :return: numpy-array of the Gillespie-evolution as well as a list of generated time points.
    """
    if init_seq is None:
        seq = np.array([np.random.randint(2, size=L)])
    else:
        seq = init_seq

    t = [0]
    m = []

    # We exclude the neighbour interactions for both sites having the same methylation state
    k3 = a * s  # demethylation due to neighbouring interaction
    k4 = a * s * y  # methylation due to neighbouring interaction
    k2 = s * y  # spontaneous methylation
    k1 = s  # spontaneous demethylation

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

    while t[-1] < tend:
        current_seq = seq[-1]
        # Update number of methy. and unmethyl. states and the two-sites states.
        counter0 = len(np.where(current_seq == 0)[0])
        counter1 = len(np.where(current_seq == 1)[0])

        m.append(counter1)
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
        t.append(t[-1] + tau)
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
        seq = np.append(seq, [current_seq], axis=0)

    m.append(len(np.where(seq[-1] == 1)[0]))
    return seq, t, m


def both_neighbours(init_seq=None, tend=50, a=1.25, s=1.47e-3, y=0.3, L=100):
    """
    Runs the Gillespie algorithm accounting for both left and right nearest neighbours
    :param init_seq: initial sequence. Has the form np.array([[0, 1, ..., 1]])
    :param tend: stopping time
    :param a: parameter value that indicates the strength of neighbouring interaction, a>1
    :param s: demethylation rate
    :param y: strength of methylation vs. demethylation, y<1: demethylated state preferred state
    :return: numpy-array of the Gillespie-evolution as well as a list of generated time points.
    """
    if init_seq is None:
        seq = np.array([np.random.randint(2, size=L)])
    else:
        seq = init_seq

    t = [0]
    m = []

    # We exclude the neighbour interactions for both sites having the same methylation state

    k3 = a * s  # demethylation due to neighbouring interaction
    k4 = a * s * y  # methylation due to neighbouring interaction
    k2 = s * y  # spontaneous methylation
    k1 = s  # spontaneous demethylation

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

    while t[-1] < tend:
        current_seq = seq[-1]
        # Update number of methy. and unmethyl. states and the two-sites states.
        counter0 = len(np.where(current_seq == 0)[0])
        counter1 = len(np.where(current_seq == 1)[0])

        m.append(counter1)
        # counter_xyx checks the number of triplets that are in state xyx, where y denotes the state of X_n and x
        # the state of the left and the right neighbour. Note that we assume the relevance of neighbouring
        # interaction only when both neighbours have the same state that is different to X_n

        counter010 = 0
        counter101 = 0
        for n in range(current_seq.shape[0]):
            if current_seq[n] == 0:
                if current_seq[neighbour_l[n]] == 1 and current_seq[neighbour_r[n]] == 1:
                    counter101 += 1
            else:
                if current_seq[neighbour_l[n]] == 0 and current_seq[neighbour_r[n]] == 0:
                    counter010 += 1

        r3 = counter010 * k3
        r4 = counter101 * k4
        r2 = counter0 * k2
        r1 = counter1 * k1
        # Update all possible rates
        possible_rates_total = [r3, r4, r2, r1]
        rate_sum = sum(possible_rates_total)
        # randomly choose next timepoint for an event to happen from an exponential distribution with mean
        # rate_sum
        tau = np.random.exponential(scale=(1 / rate_sum))
        t.append(t[-1] + tau)
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
            # check which X_n = 1 has neighbours X_{n-1} = X_{n+1} = 0
            suitable_n = []
            for ind in n[0]:
                if current_seq[neighbour_l[ind]] == 0 and current_seq[neighbour_r[ind]] == 0:
                    suitable_n.append(ind)
            chosen_n = np.random.randint(0, len(suitable_n))
            current_seq[suitable_n[chosen_n]] = 0
        # neighboring interaction methylation for X_n = 0, X_{n-1} = 1
        else:
            n = np.where(current_seq == 0)
            # check which X_n = 0 has neighbours X_{n-1} = X_{n+1} = 1
            suitable_n = []
            for ind in n[0]:
                if current_seq[neighbour_l[ind]] == 1 and current_seq[neighbour_r[ind]] == 1:
                    suitable_n.append(ind)
            chosen_n = np.random.randint(0, len(suitable_n))
            current_seq[suitable_n[chosen_n]] = 1
        seq = np.append(seq, [current_seq], axis=0)

    m.append(len(np.where(seq[-1] == 1)[0]))
    return seq, t, m
