"""
Fitting a function to simulated data described in section 3.3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import Model
from AverageMethylatedSites import average_methylated_sites_multirun

# Parameter values
runs = 30  # Number of runs
L = 200
tend = 5000
s = 1.47e-3

# function to fit
def exp_m(t, b, beta, c):
    return b - b * np.exp(-beta * t) + c * np.exp(-beta * t)


# initial guess for curve fit coefficients
b0 = 0.2
beta0 = np.log(1.002)
c0 = 1/2

for y in [0.1, 0.3, 0.6, 0.9]:
    for a in [0.1, 0.5, 1, 2, 10]:
        all_mlists = [0] * runs
        all_tlists = [0] * runs
        all_mlists_freq = [0] * runs

        for n in range(runs):
            seq, t, m = Model.one_neighbour(init_seq=None, tend=tend,
                                               a=a, s=s, y=y, L=L)
            all_tlists[n] = t
            all_mlists[n] = m
            all_mlists_freq[n] = np.divide(m, L)

        common_timepoints, m_list, u_list = average_methylated_sites_multirun(all_mlists=all_mlists,
                                                                              all_tlists=all_tlists,
                                                                              L=L, tend=tend)
        m_list_freq = np.divide(m_list, L)
        parameters_m, covariance_m = curve_fit(exp_m, common_timepoints, m_list_freq, p0=(b0, beta0, c0))

        text = f"""
        {'-' * 40}
        # Fitted values:
        # y: {str(y)}
        # a: {str(a)}
        # Parameter estimates: {parameters_m}
        # Variance of parameter estimates: {covariance_m}

        {'-' * 40}
        """

        print(text)

        fig, ax = plt.subplots()
        # plot of the first run
        ax.plot(all_tlists[0], all_mlists_freq[0], color='#003cb3', alpha=0.5)
        # plot of the second run
        ax.plot(all_tlists[1], all_mlists_freq[1], color='#003cb3', alpha=0.5)
        ax.plot(all_tlists[2], all_mlists_freq[2], color='#003cb3', alpha=0.5)
        # plot of the computed averaged data
        ax.plot(common_timepoints, m_list_freq, color='#003cb3', label='averaged methylation level')
        # plot fitted curve methylation level
        ax.plot(common_timepoints, exp_m(common_timepoints, *parameters_m), color='#7cfc00', linestyle='-',
                label='mean methylation level fit: b=%5.3f, beta=%5.3f, c=%5.3f' % tuple(parameters_m))
        plt.title('Mean methylation level for multiple runs (y=' + str(y) + ' a=' + str(a) + ')')
        plt.xlabel('time')
        plt.ylabel('proportion of CpG sites')
        ax.legend()

plt.show()
