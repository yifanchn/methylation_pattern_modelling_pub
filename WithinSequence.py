"""
The following function is the simulation of our within sequence interactions as described in Model.py
until the end of one predefined generation.
"""

import numpy as np

def withinsequence(seq, t, L, m, a=1.25, s=1.47e-3, y=0.3, tgen=50):
    """
    Simulation of the within sequence changes until the end of one generation
    :param seq: initial sequence which will be updated after each event happens
    :param t: time sequence
    :param L: Number of CpG sites for sequence
    :param m: sequence containing the methylation level
    :param a: or alpha, indicating the strength of neighbouring interactions
    :param s: demethylation rate
    :param y: strength of methylation vs demethylation rate, y<1: demethylation prefered
    :param tgen: duration of one generation
    :return: array of methylome evolution of the input init_seq until the end of one generation t_gen
    """
    # Reaction rates k1-k4
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

    has_tgen_reached: list[bool] = [False]

    while not has_tgen_reached[-1]:
        current_seq = seq[-1]

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
        if t[-1] + tau > tgen:
            has_tgen_reached.append(True)
        else:
            has_tgen_reached.append(False)
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

            m.append(len(np.where(current_seq == 1)[0]))
            seq = np.append(seq, [current_seq], axis=0)
    return seq, has_tgen_reached, t, m
