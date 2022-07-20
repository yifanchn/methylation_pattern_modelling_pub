"""
This is the module that calculates the number of (un-)methylated sites from t=0 to t=tend, averaged over a certain
number of runs.
The runs have already taken place and their lists m and t are saved so that you can enter them as
variables. Suppose you performed n runs of the new model, and m_i and t_i are the lists that correspond to run i < n.
Then you can define the variables as follows:
all_mlists = [m_1, m_2, ..., m_n]
all_tlists = [t_1, ..., t_n]
"""
import numpy as np
from bisect import bisect_right
from matplotlib import pyplot as plt


def average_methylated_sites_multirun(all_mlists, all_tlists, L, tend):
    """
    Computes the multiple-run average of (un-)methylated sites for normed time points in the range 0-tend.
    :param all_mlists: see module description
    :param all_tlists: see module description
    :param L: length of the sequences (should be equal for all runs)
    :param tend: the usual endpoint tend
    :return: common_ts is the list with the common timepoints of all runs, average_methylated and
    average_unmethylated are the corresponding averaged numbers of (un-)methylated sites
    """
    number_of_runs = len(all_mlists)
    # list for the length of all time-lists, needed for the definition of the shortest time-list
    collection_of_tlengths = []
    # define a list of all lists of unmethylated sites
    all_ulists = []
    for run in range(number_of_runs):
        temp_ulist = []
        collection_of_tlengths.append(len(all_tlists[run]))
        current_mlist = all_mlists[run]
        for m in current_mlist:
            temp_ulist.append(L-m)
        all_ulists.append(temp_ulist)
    # get a list for the time-points from which we can then create averages - the "common time points"
    common_ts = np.linspace(0, tend, num=min(collection_of_tlengths))
    collection_of_unmethylated_sites = []
    collection_of_methylated_sites = []
    for run in range(number_of_runs):
        unmethylated_sites = []
        methylated_sites = []
        tlist = all_tlists[run]
        for t in common_ts:
            # choose the time points in the time-list for this particular run that are closest to t from the list of
            # common time points
            right = bisect_right(tlist, t)
            tind = right - 1
            # get the numbers of (un-)methylated sites at time tind for this particular run
            temp_status = [all_ulists[run][tind], all_mlists[run][tind]]
            unmethylated_sites.append(temp_status[0])
            methylated_sites.append(temp_status[1])
        collection_of_unmethylated_sites.append(unmethylated_sites)
        collection_of_methylated_sites.append(methylated_sites)
    # averaging the data - so far, collection_of_(un-)methylated_sites contain the "time point-normed" data for each of
    # the runs in a separate list, but we want a single list with the averages
    average_unmethylated = []
    average_methylated = []
    for timepoint in range(len(common_ts)):
        unmeth_sum = 0
        meth_sum = 0
        for run in range(number_of_runs):
            unmeth_sum += collection_of_unmethylated_sites[run][timepoint]
            meth_sum += collection_of_methylated_sites[run][timepoint]
        average_unmethylated.append(unmeth_sum / number_of_runs)
        average_methylated.append(meth_sum / number_of_runs)
    return common_ts, average_methylated, average_unmethylated


# EXAMPLE USAGE OF average_methylated_sites_multirun()
if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # set-up the example
    L_example = 10
    tend_example = 8.0
    # the exemplary methylation- and time-lists have length 11 and 13
    all_mlists_example = [[5, 4, 5, 4, 3, 2, 1, 2, 1, 2, 3],
                          [5, 4, 3, 2, 3, 4, 3, 2, 1, 0, 1, 0, 1]]
    all_tlists_example = [[0.0, 1.1, 2.0, 2.3, 3.1, 4.0, 4.5, 5.7, 6.8, 7.0, 7.2],
                          [0.0, 1.5, 2.2, 2.8, 4.0, 4.7, 5.1, 5.4, 6.1, 6.2, 6.6, 7.0, 7.4]]
    common_timepoints, m_list, u_list = average_methylated_sites_multirun(all_mlists_example, all_tlists_example,
                                                                          L_example, tend_example)
    # ------------------------------------------------------------------------------------------------------------------
    # plot the example
    fig, ax = plt.subplots()
    # plot of the first example run
    ax.plot(all_tlists_example[0], all_mlists_example[0], color='#003cb3', alpha=0.5)
    ax.plot(all_tlists_example[0], [L_example - x for x in all_mlists_example[0]], color='#6699ff', alpha=0.5)
    # plot of the second example run
    ax.plot(all_tlists_example[1], all_mlists_example[1], color='#003cb3', alpha=0.5)
    ax.plot(all_tlists_example[1], [L_example - x for x in all_mlists_example[1]], color='#6699ff', alpha=0.5)
    # plot of the computed averaged data
    ax.plot(common_timepoints, u_list, color='#6699ff', label='averaged no. unmethylated')
    ax.plot(common_timepoints, m_list, color='#003cb3', label='averaged no. methylated')
    plt.title('Averaging the number of (un-)methylated sites for multiple runs (example data)')
    plt.xlabel('time')
    plt.ylabel('number of CpG sites')
    ax.legend()
    plt.show()