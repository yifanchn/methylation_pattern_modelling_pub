# modules from this repository:
import Model

# general imports:
import numpy as np


def recombination(X, num_parents=2, len_recom=5, alternating=True):
    """
    Recombines the DNA of several individuals according to the with-linkage inheritance.
    :param X: the list with all the other individuals' sequences from generation m, of the form
    [np.array([['X111', 'X112', 'X113'], ['X121', 'X122', 'X123']]), ..., np.array([['X311', 'X312', 'X313'],
    ['X321', 'X322', 'X323']])]. The last available sequence per individual is used (the one that represents the current
    status of the methylome).
    :param num_parents: number of parents that the new individual will have.
    :param len_recom: length of the DNA snippets that should stay together during recombination, has to be a divisor of
    the sequence length N.
    :param alternating: if set True, will recombine strictly alternating snippets.
    :return: a numpy-array with the first sequence of the new child.
    """
    if len(X[0][0]) % len_recom != 0:
        print('Choose len_recom differently! N = ' + str(len(X[0][0])) + ' has to be a multiple of len_recom.')
        return
    else:
        num_recom = len(X[0][0]) / len_recom
        # randomly selecting the parents from the arrays in X
        parent_indices = np.random.default_rng().choice(len(X), size=num_parents, replace=False)
        # print('The parents are:')
        # getting a list of the possible DNA-snippets
        dna_snippets = []
        for par_ind in parent_indices:
            # print('individual no. ' + str(par_ind + 1) + ' with final sequence ' + str(X[par_ind][-1]))
            dna_snippets.append(np.split(X[par_ind][-1], num_recom))
        # recombine the snippets randomly
        child = ()
        if alternating:
            random_order = np.random.randint(num_parents, size=1)
            for n in range(int(num_recom)):
                chosen_parent = (n + random_order[0]) % num_parents
                child += (dna_snippets[chosen_parent][n],)
        else:
            for n in range(int(num_recom)):
                chosen_parent = np.random.randint(num_parents, size=1)
                child += (dna_snippets[chosen_parent[0]][n],)
        child = np.concatenate(child, axis=0)
        # print(child)
    return np.array([child])


# exemplary usage
if __name__ == "__main__":
    # PARAMETER DEFINITION ---------------------------------------------------------------------------------------------
    # number of individuals per generation
    generation_size = 100
    # number of generations -- this kind of inheritance is implemented a little bit differently
    # than the no-linkage-inheritance: here, you don't select the final time point and see how many generations "fit in"
    # but you rather determine the straight-forward number of generations you want to simulate
    G = 50
    # tend in this context is then defined as the maximum time unit up to which each generation can live, not the final
    # time point at which the whole between sequence interaction simulation stops
    tend = 1000
    # length of the snippets for recombination
    snippet_length = 50
    # sequence length
    L = 200
    # spontaneous methylation rate
    s = 1.47e-3
    # strength of neighbouring interactions
    a = 1
    # favoured reaction parameters
    y = 2

    # SET UP FIRST GENERATION ------------------------------------------------------------------------------------------
    # random initial cohort
    C = []
    for new_seq in range(generation_size):
        C.append(np.array([np.random.randint(2, size=L)]))

    # executing Gillespie for all members of generation 0
    for ind_seq in range(generation_size):
        # one neighbour
        gillespie, ts, ms = Model.one_neighbour(init_seq=C[ind_seq], tend=tend, a=a, s1=s, s2=s, y=y, L=L)

        # both neighbours
        # gillespie, ts, ms = Model.both_neighbours(init_seq=C[ind_seq], tend=tend, a=a, s1=s, s2=s, y=y, L=L)

        C[ind_seq] = np.append(C[ind_seq], gillespie, axis=0)

    # BETWEEN SEQUENCE AND WITHIN SEQUENCE INTERACTIONS ----------------------------------------------------------------
    # generating children in G generations (the last generation is the G'th generation) and running Gillespie on them
    X_cohort = C
    for gen in range(G):
        X_cohort_next_gen = []
        for new_seq in range(generation_size):
            X_cohort_next_gen.append(recombination(X_cohort, num_parents=2, len_recom=snippet_length, alternating=True))

        # executing Gillespie for all members of generation gen+1
        for ind_seq in range(len(X_cohort_next_gen)):
            # one neighbour
            gillespie, ts, ms = Model.one_neighbour(init_seq=X_cohort_next_gen[ind_seq], tend=tend, a=a, s1=s, s2=s,
                                                    y=y, L=L)
            # both neighbours
            # gillespie, ts, ms = Model.both_neighbours(init_seq=X_cohort_next_gen[ind_seq], tend=tend, a=a, s1=s,
            #                                              s2=s, y=y, L=L)
            X_cohort_next_gen[ind_seq] = np.append(X_cohort_next_gen[ind_seq], gillespie, axis=0)

        # HERE is the place to write code that is supposed to analyze e. g. cluster sizes or the mSFS of the current
        # generation. It is saved in the variable X_cohort_next_gen.

        # overriding X_cohort to be the basis of the next generation
        X_cohort = X_cohort_next_gen

        print("Done with generation no. " + str(gen + 1))
