import numpy as np
from scipy.stats import bootstrap, permutation_test, spearmanr, pearsonr, kendalltau

#### Calculate real correlation
def linear_corr (x, y, corr_method='spearman'):
    if corr_method == 'spearman':
        r, p = spearmanr(x, y)
    elif corr_method == 'pearson':
        r, p = pearsonr(x, y)
    elif corr_method == 'kendall':
        r, p = kendalltau(x, y)
    return r, p

#### BOOTSTRAP to estimate the CI 
def bootstrap_corr (x, y, corr_method = 'spearman', n_boot = 1000, seed = 1979):

    real_r, _ = linear_corr(x,y,corr_method)

    boot_res = bootstrap(
                        data=(x, y), 
                        statistic=lambda x, y: linear_corr(x,y,corr_method)[0],
                        method="bca", 
                        vectorized=False, 
                        n_resamples=n_boot, 
                        paired=True, 
                        random_state=seed
                        )
    
    ci =[round(boot_res.confidence_interval[0],2), 
        round(boot_res.confidence_interval[1], 2)]
    
    # get the distribution of correlation r
    all_rs = boot_res.bootstrap_distribution

    # Determine the P-value from the distribution of correlation coefficient r
    # case : two.sided (from boot_cor_test function of the package TOSTER in R)
    phat = (sum(all_rs < 0) + 0.5 * sum(all_rs == 0))/n_boot
    p = 2 * min(phat, 1 - phat)

    return real_r, p, ci, all_rs

#### PERMUTATION test to test if the correlation is true (p-value) 
# The results of permutation_test is very similar to the result obtain with bootstrap
def permutation_corr (x, y, corr_method = 'spearman', n_boot = 1000, seed = 1979):

    real_r, _ = linear_corr(x,y,corr_method)

    perm_res = permutation_test(
                data=(x, y), 
                statistic=lambda x, y: linear_corr(x,y,corr_method)[0],
                permutation_type="pairings", 
                n_resamples=n_boot, 
                random_state=seed
                )

    # P-value determine by the function permutation_test
    # p = perm_res.pvalue
    
    # Determine the P-value frome the null distribution
    null_rs = perm_res.null_distribution
    better_null_ixs = np.where(np.abs(null_rs) > np.abs(real_r))[0]
    p = (float(len(better_null_ixs))/float(len(null_rs))) 

    return real_r, p, null_rs
