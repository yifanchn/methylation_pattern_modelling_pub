# modules from this repository:
import Model

# general imports:
import os  # only necessary if one wants to save results by setting save=True in the function
# average_cluster_size_multirun()-- can lead to problems with some operating systems and can be
# commented if that is the case
import numpy as np
from bisect import bisect_right
from statistics import fmean
import matplotlib.pyplot as plt
import scipy.stats as stats


def average_cluster_size(seq, tlist, t):
    """
    Computes the average number of sites that are involved in a (un-)methylated cluster around a specific time point.
    :param seq: the numpy-array of the DNA sequence
    :param tlist: the list of time points generated by Gillespie
    :param t: the desired time point
    :return: list of the average size of the unmethylated as well as methylated clusters
    """
    # choose the time points in the time-list that are closest to t
    right = bisect_right(tlist, t)
    if right == 0:
        print('The value for t is lower than the first time point. Lowest option for t: t = ' + str(tlist[0]) +
              ', your choice of t: t = ' + str(t))
        return
    elif right > len(tlist):
        print('The value for t exceeds the list of possible time points. Highest option for t: t = ' + str(tlist[-1]) +
              ', your choice of t: t = ' + str(t))
        return
    # we will continue with the lower bound (but of course, one could also do it with the upper one)
    tind = right - 1
    # seqt is the sequence that was created at time lowert
    seqt = seq[tind]
    # get the number of interfaces in seq (inter_counter is the same as number of clusters)
    inter_counter = 0
    previous = seqt[-1]
    for ind in seqt:
        if ind != previous:
            inter_counter += 1
        previous = ind
    if inter_counter != 0:
        ones = sum(seqt)
        average_cluster1 = ones / (0.5*inter_counter)
        zeros = len(seqt) - ones
        average_cluster0 = zeros / (0.5*inter_counter)
    # case where seqt = [0, 0, ..., 0] or seqt = [1, 1, ..., 1]
    else:
        if seqt[0] == 0:
            average_cluster1 = 0
            average_cluster0 = len(seqt)
        else:
            average_cluster1 = len(seqt)
            average_cluster0 = 0
    return [average_cluster0, average_cluster1]


def average_cluster_size_single_run(seq, t):
    """
    Calculates the mean cluster size (methylated and unmethylated) for one single run over all time points.
    :param seq: the sequence of the form np.array([[0, 1, ..., 1], ..., [1, 1, ..., 0]])
    :param t: the list of time points
    :return: average cluster size unmethylated, average cluster size methylated
    """
    unmethylated_clusters = []
    methylated_clusters = []
    for timepoint in t:
        temp_cluster = average_cluster_size(seq, t, timepoint)
        unmethylated_clusters.append(temp_cluster[0])
        methylated_clusters.append(temp_cluster[1])
    return fmean(unmethylated_clusters), fmean(methylated_clusters)


def average_cluster_size_multirun(num_runs, tend, a, s, y, L, with_plot=True, save=False):
    """
    Calculates the average cluster size of methylated and unmethylated sites over multiple runs. Also plots the results
    if desired.
    :param num_runs: number of runs over which the mean should be calculated.
    :param tend: point in time when the algorithm is supposed to stop.
    :param a: scaling factor, 0 < a < 0.5.
    :param s: demethylation rate
    :param y: strength of methylation vs. demethylation. y<1: demethylated state is preferred.
    :param L: sequence length
    :param with_plot: plots the cluster sizes if set True.
    :param save: if set True saves the results of the simulation for further analysis.
    :return: mean of methylated and demethylated cluster size after num_runs runs.
    """
    unmethylated_means = []
    methylated_means = []
    all_runs_unmethylated_clusters = []
    all_runs_methylated_clusters = []
    collection_of_tlengths = []
    if with_plot:
        fig, ax = plt.subplots()
    for run in range(num_runs):

        # SET THE DESIRED MODEL HERE
        # one neighbour
        seq, t, m = Model.one_neighbour(tend=tend, a=a, s1=s, s2=s, y=y, L=L)
        # both neighbours
        # seq, t, m = Model.both_neighbours(tend=tend, a=a, s1=s, s2=s, y=y, L=L)

        # saving the result of the simulation for further analysis
        if save:
            filename = 'avg_cluster_sizes_run_no_' + str(run) + '_param_a_' + str(a) + '_param_y_' + str(y)
            if os.path.exists('DATA/'):
                if m is None:
                    np.savez('DATA/{}.npz'.format(filename), seq=seq, t=t)
                else:
                    np.savez('DATA/{}.npz'.format(filename), seq=seq, t=t, m=m)
            else:
                print('Please create a new folder named "DATA".')

        # lists of the average cluster sizes at each time point in t
        unmethylated_clusters = []
        methylated_clusters = []
        for time in t:
            temp_cluster = average_cluster_size(seq, t, time)
            unmethylated_clusters.append(temp_cluster[0])
            methylated_clusters.append(temp_cluster[1])
        # lists of the form [[(un-)methylated cluster sizes for run 0], ..., [(un-)methylated cluster sizes for run 99]]
        all_runs_unmethylated_clusters.append(unmethylated_clusters)
        all_runs_methylated_clusters.append(methylated_clusters)
        # same as in StatisticAnalysis
        collection_of_tlengths.append(len(t))
        if with_plot:
            # transparency of the plotted lines; alpha=1 means no transparency, alpha=0 is invisible. The oldest plot
            # is the most visible.
            alpha = 1 / (run + 1)
            ax.plot(t, unmethylated_clusters, color='#6699ff', alpha=alpha)
            ax.plot(t, methylated_clusters, color='#003cb3', alpha=alpha)
        # for the sake of less computational expenses, we only save the means and override the lists in the next
        # iteration
        unmethylated_means.append(fmean(unmethylated_clusters))
        methylated_means.append(fmean(methylated_clusters))
    mean_unmethylated = fmean(unmethylated_means)
    mean_methylated = fmean(methylated_means)
    # generating the mean
    common_ts = np.linspace(0, tend, num=min(collection_of_tlengths))
    all_runs_averaged_unmethylated = []
    all_runs_averaged_methylated = []
    for timepoint in range(len(common_ts)):
        unmeth_sum = 0
        meth_sum = 0
        for run in range(num_runs):
            unmeth_sum += all_runs_unmethylated_clusters[run][timepoint]
            meth_sum += all_runs_methylated_clusters[run][timepoint]
        all_runs_averaged_unmethylated.append(unmeth_sum / num_runs)
        all_runs_averaged_methylated.append(meth_sum / num_runs)
    if with_plot:
        label1 = 'demethylated cluster mean = ' + str(round(mean_unmethylated, 2))
        label2 = 'methylated cluster mean = ' + str(round(mean_methylated, 2))
        plt.plot(common_ts, all_runs_averaged_unmethylated, color='red', label=label1)
        plt.plot(common_ts, all_runs_averaged_methylated, color='yellow', label=label2)
        plt.title('Evolution of cluster size for multiple sequences')
        plt.xlabel('time')
        plt.ylabel('average cluster size')
        ax.legend()
        plt.show()
    return [mean_unmethylated, mean_methylated]


def cluster_analysis_saved(a, y):
    """
    Calculating the mean cluster sizes for each of the saved runs with the given input parameters. Requires that the
    file exists.
    :param a: a
    :param y: y
    :return: list of average cluster sizes for the unmethylated cluster means, the methylated ones and the filename
    where this data has been saved.
    """
    filename_means = 'mean_of_each_run_param_a_' + str(a) + '_param_y_' + str(y)
    list_unmethylated = []
    list_methylated = []
    for run in range(100):
        filename = 'avg_cluster_sizes_run_no_' + str(run) + '_param_a_' + str(a) + '_param_y_' + str(y)
        npzfile = np.load('DATA/{}.npz'.format(filename))
        seq, t = npzfile['seq'], npzfile['t']
        mean_unmeth, mean_meth = average_cluster_size_single_run(seq, t)
        list_unmethylated.append(mean_unmeth)
        list_methylated.append(mean_meth)
    return list_unmethylated, list_methylated, filename_means


def two_sample_t_test_average_cluster_size(means_one_neighbor, means_both):
    """
    Performs a two-sample t-test on the means of the cluster sizes. ATTENTION: you can only compare demethylated to
    demethylated (or methylated to methylated) and a and y have to be the same!
    :param means_one_neighbor: numpy-array of the mean cluster size from the run with one neighbor as influence factor
    :param means_both: numpy-array of the mean cluster size from the run with two neighbors as influence factor
    :return:
    """
    # calculate the variances since we need to decide if we’ll assume the two populations have equal variances or not.
    # As a rule of thumb, we can assume the populations have equal variances if the ratio of the larger sample variance
    # to the smaller sample variance is less than 4:1. (source: https://www.statology.org/two-sample-t-test-python/)
    var_one_neighbor = np.var(means_one_neighbor)
    var_both = np.var(means_both)
    equal_var = False
    if var_one_neighbor > var_both:
        if var_one_neighbor / var_both < 4:
            equal_var = True
    else:
        if var_both / var_one_neighbor < 4:
            equal_var = True
    # perform the test based on the result of the variance-checking
    return stats.ttest_ind(a=means_one_neighbor, b=means_both, equal_var=equal_var)


def wilcoxon_ranksum(means_one_neighbor, means_both):
    """
    For the samples where the test for normal distribution failed: Performs a Wilcoxon rank-sum test on the means of the
    cluster sizes. ATTENTION: you can only compare unmethylated to unmethylated (or methylated to methylated) and a and
    y have to be the same!
    :param means_one_neighbor: numpy-array of the mean cluster size from the run with one neighbor as influence factor
    :param means_both: numpy-array of the mean cluster size from the run with two neighbors as influence factor
    :return: corresponding p-value
    """
    statistic, p = stats.ranksums(means_one_neighbor, means_both)
    return p


def test_for_normal_distribution(means):
    """
    Performs a test for normal distribution with the means of one parameter combination
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html).
    :param means: numpy-array of the mean cluster size from all runs with the same parameter combination and either one
    or both neighbours
    :return:
    """
    k2, p = stats.normaltest(means)
    return p


def histogram_average_cluster_sizes(means_one_neighbor, means_both, a, y, status):
    """
    Plots the distribution of the cluster sizes of both models. ATTENTION: you can only compare unmethylated to
    unmethylated (or methylated to methylated) and a and y have to be the same!
    :param means_one_neighbor: numpy-array of the mean cluster size from the run with one neighbor as influence factor
    :param means_both: numpy-array of the mean cluster size from the run with two neighbors as influence factor
    :param a: parameter value of a
    :param y: parameter value of y
    :param status: methylation status
    :return:
    """
    if status == 'unmethylated':
        col = '#6699ff'
    else:
        col = '#003cb3'
    n_bins = 20
    mean_one_neighbor = np.mean(means_one_neighbor)
    sd_one_neighbor = np.std(means_one_neighbor)
    mean_two_neighbors = np.mean(means_both)
    sd_two_neighbors = np.std(means_both)

    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
    count, bins, ignored = axs[0].hist(means_one_neighbor, bins=n_bins, density=True, color=col)
    # plotting the probability density function for a normal distribution with the same mean and standard deviation
    mu = mean_one_neighbor
    sigma = sd_one_neighbor
    axs[0].plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), color='r')
    # insert mean
    axs[0].axvline(mean_one_neighbor, color='#002b80', label='mean = ' + str(round(mean_one_neighbor, 2)))
    # insert standard deviation
    axs[0].axvline(mean_one_neighbor - sd_one_neighbor, color='#666699', linestyle='dashed',
                   label='sd = ' + str(round(sd_one_neighbor, 2)))
    axs[0].axvline(mean_one_neighbor + sd_one_neighbor, color='#666699', linestyle='dashed')
    axs[0].set_title('One neighbour')
    axs[0].legend()

    count, bins, ignored = axs[1].hist(means_both, bins=n_bins, density=True, color=col)
    # plotting the probability density function for a normal distribution with the same mean and standard deviation
    mu = mean_two_neighbors
    sigma = sd_two_neighbors
    axs[1].plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), color='r')
    # insert mean
    axs[1].axvline(mean_two_neighbors, color='#002b80', label='mean = ' + str(round(mean_two_neighbors, 2)))
    # insert standard deviation
    axs[1].axvline(mean_two_neighbors - sd_two_neighbors, color='#666699', linestyle='dashed',
                   label='sd = ' + str(round(sd_two_neighbors, 2)))
    axs[1].axvline(mean_two_neighbors + sd_two_neighbors, color='#666699', linestyle='dashed')
    axs[1].set_title('Both neighbours')
    axs[1].legend()
    fig.suptitle('a = ' + str(a) + ', y = ' + str(y) + ', ' + str(status) + ' clusters')

    # filename for the NEW MODEL
    filename = 'NEW_MODEL_histogram_a_i_' + str(a) + '_param_y_' + str(y)
    plt.savefig('PLOTS/{}.png'.format(filename), bbox_inches='tight')

    #plt.show()
