# modules from this repository:
import Model

# general imports:
import numpy as np
import matplotlib.pyplot as plt


def mSFS_custom_sequences(sequences, cohort=False, absolute=False):
    """
    Plots the mSFS similarly to mSFS_multirun, but does not execute the Gillespie algorithm s. t. the sequences are
    handled in their original state.
    :param cohort: set True if the list of sequences is of the form [np.array([['X111', 'X112', 'X113'],
    ['X121', 'X122', 'X123']]), ..., np.array([['X311', 'X312', 'X313'], ['X321', 'X322', 'X323']])].
    :param absolute: decides whether the plot shows absolute or relative values.
    :param sequences: list of sequences that one wants to calculate the mSFS for. All must have the same length!
    :return: nothing, shows the plot.
    """
    if cohort:
        seq_length = len(sequences[0][0])
        unmethylated_at_index = []
        for ind in range(seq_length):
            num_unmethylated = 0
            for seq in sequences:
                if seq[-1][ind] == 0:
                    num_unmethylated += 1
            unmethylated_at_index.append(num_unmethylated)
    else:
        seq_length = len(sequences[0])
        unmethylated_at_index = []
        for ind in range(seq_length):
            num_unmethylated = 0
            for seq in sequences:
                if seq[ind] == 0:
                    num_unmethylated += 1
            unmethylated_at_index.append(num_unmethylated)
    max_unmethylated = max(unmethylated_at_index)
    occurences = []
    for i in range(max_unmethylated + 1):
        if absolute:
            occurences.append(unmethylated_at_index.count(i))
        else:
            occurences.append(unmethylated_at_index.count(i) / seq_length)
    length_discrepancy = (len(sequences)+1) - len(occurences)
    if length_discrepancy != 0:
        for i in range(length_discrepancy):
            occurences.append(0)
    fig, ax = plt.subplots(1, 1)
    axis = np.linspace(0, len(sequences), num=len(sequences)+1)
    ax.plot(axis, occurences, 'o-', color='#6699ff', label='simulation')
    ax.set_xlim([0, len(sequences)])
    fig.suptitle('Methylation site frequency spectrum')
    ax.set_xlabel('Number of demethylated sites')
    if absolute:
        ax.set_ylabel('Number of CpG sites')
    else:
        ax.set_ylabel('Proportion of CpG sites')
    plt.show()


# exemplary usage
if __name__ == "__main__":
    # PARAMETER DEFINITION ---------------------------------------------------------------------------------------------
    # number of individuals
    N = 100
    # end point
    tend = 1000
    # sequence length
    L = 200
    # spontaneous demethylation rate
    s = 1.47e-3
    # strength of neighbouring interactions
    a = 1
    # favoured reaction parameters
    y = 2

    # SET UP AND PLOT mSFS ---------------------------------------------------------------------------------------------
    # initial cohort
    C = []
    for new_seq in range(N):
        C.append(np.array([np.random.randint(2, size=L)]))

    # plot mSFS before Gillespie
    mSFS_custom_sequences(C, cohort=True, absolute=True)

    # executing Gillespie for all members of the cohort
    for ind_seq in range(N):
        # one neighbour
        gillespie, ts, ms = Model.one_neighbour(init_seq=C[ind_seq], tend=tend, a=a, s1=s, s2=s, y=y, L=L)
        # both neighbours
        # gillespie, ts, ms = Model.both_neighbours(init_seq=C[ind_seq], tend=tend, a=a, s1=s, s2=s, y=y, L=L)
        C[ind_seq] = np.append(C[ind_seq], gillespie, axis=0)

    # plot mSFS after Gillespie
    mSFS_custom_sequences(C, cohort=True, absolute=True)
