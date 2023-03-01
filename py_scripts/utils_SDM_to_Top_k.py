import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import pandas as pd
import time
import scipy.integrate as integrate
from types import SimpleNamespace
import seaborn as sns
import pandas as pd
from scipy.integrate import quad
from scipy.special import comb

# All taken from SDM project. "Attention Approximates Sparse Distributed Memory". 

def space_frac_to_hamm_dist(n, space_frac_rang):
    """ Computes the Hamming distance that should be used for a circle 
    to have an area that includes a given fraction of a given n 
    dimensional space.
    
    args::
    - n = space dimension
    - space_frac_rang = list of space fractions to use
    
    returns::
    -list of hamming distances to use
    """
    
    hamm_distances = []
    for space_frac in space_frac_rang:
        hamm_distances.append( int(binom.ppf(space_frac, n, 0.5)) )
    return hamm_distances

def hamm_dist_to_space_frac(n, hamm_dist_rang):
    """ Computes the space fraction $p$ that corresponds to a given Hamming distance input
    
    args::
    - n = space dimension
    - space_frac_rang = list of Hamming distances used
    
    returns::
    - list of p fractions
    """
    
    pfracs = []
    for hd in hamm_dist_rang:
        pfracs.append( binom.cdf(hd, n, 0.5) )
    return pfracs

def SNR_optimal(m,r,n):
    optimal_p = 1/(2*m*r)**(1/3)
    d = space_frac_to_hamm_dist(n, [optimal_p])[0]
    print('Optimal p', optimal_p)
    print('Optimal d', d)
    return optimal_p, d

def memory_derivative(z,r):
    return 1/2*(z**4/(2*r**4*z**2 + r**3*z**6 + 2*np.sqrt(r**8*z**4 + r**7*z**8))**(1/3) + (2*r**4*z**2+ r**3*z**6 + 2*np.sqrt(r**8*z**4 + r**7 *z**8))**(1/3)/r**2 + z**2/r)

def Memory_Capacity_optimal(m,r,n):
    # z_score so the overall retrieval is 99%
    prob_retrieval_of_whole_pattern = 0.5
    prob_per_bit = prob_retrieval_of_whole_pattern**(1/n)
    z_score_per_bit = norm.ppf(prob_per_bit)
    print('z score per bit', z_score_per_bit)
    #z = 2.33 # z-score

    optimal_p =  memory_derivative(z_score_per_bit,r)
    d = space_frac_to_hamm_dist(n, [optimal_p])[0]
    print('Optimal p', optimal_p)
    print('Optimal d', d)
    return optimal_p, d


def expected_intersection_lune(n, dvals, hamm_dist, r):
    # This equation gives the same results as the one we derive and present in the paper. It was introduced in the SDM book and runs a bit faster. 
    """
    Computes the fraction of the space that exists in the circle intersection using the Lune equation. 
    
    args::
    n = space dimension
    dvals = Hamm dist between circle centers
    hamm_dist = hamming distance radius each circle uses
    r = number of neurons
    
    hard_mem_places = turns the fraction of the space in the 
    expected number of neurons
    that exist in this fraction. 
    
    ------------
    
    returns:: 
    res = list of floats for fraction of the space
    """
    
    #ensure all are ints: 
    n = int(n)
    hamm_dist = int(hamm_dist)
    r = int(r)
    
    perc_addresses_w_neurons = r/(2**n) 
    
    res = []
    area = 0
    # compute size of circle
    for i in range(hamm_dist+1):
        area += comb(n,i)

    for d in dvals: 
        # compute lune
        d = int(d)
        lune = 0
        for i in range(d):
            j = i+1
            if j%2==0:
                continue
            lune+= comb(j-1, (j-1)/2)*comb(n-j, hamm_dist-((j-1)/2))
        intersect = area - lune
        #print(d, intersect, area, lune, perc_addresses_w_neurons)
        expected_intersect = np.log(intersect)+np.log(perc_addresses_w_neurons)
        res.append(np.exp(expected_intersect))
        
    res = np.asarray(res)
    return res

def compute_bitdist_critdist(n,hamm_dist,r,
                                m,drang_interval, print_things = False):
    
    dvals = np.arange(0,n//2, drang_interval)
    dvals = np.append(dvals, n//2) # needs to be at the end for the indifference distance calc
    target_freqs = expected_intersection_lune(n, dvals, hamm_dist, r)
    algo_max_dist = target_freqs[-1]
    sigmas = np.sqrt( target_freqs + ( (m-1)*(algo_max_dist+algo_max_dist**2) ) )

    snrs = target_freqs/sigmas

    fidelity = norm.cdf(snrs)
    bits_distances = n - (fidelity*n)
    npp = np.asarray( bits_distances - dvals )
    
    if print_things:
        print('target_fs', target_freqs)
    if (npp[~np.isnan(npp)]<0.01).all(): # if all convergence of everything then set to max value. 
        
        # if there are nans then take the last real value: 
        if np.isnan(npp).sum()>1:
            crit_dist = dvals[len(npp[~np.isnan(npp)])]
        #else take the largest value tried as the crit dist: 
        else: 
            crit_dist = dvals[-1]
    else: # take first one to cross over. 
        # we index by +3 because if it is one of the very first ones these can often be close to 0 and throw everything off. 
        crit_dist = dvals[np.argmax(npp[5:]>0)+5-1]
    
    '''if printt:
        plt.figure()
        plt.plot(dvals, bits_distances, label='Update')
        plt.plot(dvals, dvals, label='$x=y$')
        plt.show()'''
    
    # checks if the critical distance is in fact 0 as it curves the wrong way. 
    check_ind = crit_dist//(drang_interval*2)
    if bits_distances[check_ind] > dvals[check_ind]:
        crit_dist = 0
        print('critical distance plot curves wrong way thus crit dist = 0')
    
    return bits_distances, crit_dist, dvals

def Crit_Dist_optimal(m,r,n):
    drang_interval = 1
    ploteach = False

    heat = []
    crit_dists = []
    if n==64:
        hamm_distances = np.linspace(9,25, 16 ).astype(int)
    elif n==256:
        hamm_distances = np.linspace(1,125, 100 ).astype(int) 
    elif n==784:
        hamm_distances = np.linspace(300,370, 30 ).astype(int) 
    elif n==1000:
        hamm_distances = np.linspace(400,495, 95 ).astype(int)  #np.linspace(50,495, 10 ) # #np.linspace(100,495, 10 )
    elif n==3072:
        hamm_distances = np.linspace(400,1500, 50 ).astype(int)
    else: 
        raise Exception("Need to program in hamming distances")
    for hamm_dist in hamm_distances:
        hamm_dist = int(hamm_dist)
        
        bits_distances, crit_dist, dvals = compute_bitdist_critdist(n,hamm_dist,r,m,drang_interval,
                                                                print_things =False)
            
        crit_dists.append(crit_dist)
        
        if ploteach: 
            plt.plot(dvals, bits_distances, label='Update')
            plt.plot(dvals, dvals, label='$x=y$')
            plt.legend()
            plt.xlabel("Query Hamming Distance to Target")
            plt.ylabel("New Query Hamming Distance to Target")
            plt.title("hamm_dist="+str(hamm_dist)+" -SDM Query Convergence to Target Pattern")

            #plt.gcf().savefig('figures/p_'+str(p)+'SDMNewQueryConvergence.png', dpi=250)
            plt.show()

        heat.append(bits_distances)
        
    plt.plot(hamm_distances, crit_dists)
    print(hamm_distances)
    print(crit_dists)
    print(hamm_distances[np.argmax(crit_dists)])
    plt.xlabel("Hamming distance $d$")
    plt.ylabel("Critical Distance")
    #plt.axvline(447)
    plt.title("SDM Critical Distance Varying Hamming Distance $d$. \n $n=$"+str(n)+", $m=$"+str(m)+", $r=$"+str(r)) # 2^n
    #plt.gcf().savefig('figures/SDMCritDist_VaryD_N='+str(n)+'M='+str(m)+'.png', dpi=250)
    plt.show()
    opt_hamm_dist = hamm_distances[np.argmax(crit_dists)]
    return hamm_dist_to_space_frac(n, [opt_hamm_dist])[0], opt_hamm_dist  

def expected_intersection_interpretable(n, dvals, hamm_dist, r, weight_type=None, exp_beta = 0.01):
    perc_addresses_w_neurons = np.log(float(r)) - np.log(2.0**n)
    res = []
    for dval in dvals:

        possible_addresses = 0
        for a in np.arange(n-hamm_dist-(dval//2),n+0.1-dval):

            # solve just for b then c is determined. 
            bvals = np.arange(np.maximum(0,n-hamm_dist-a), dval-(n-hamm_dist-a)+0.1) # +0.1 to ensure that the value here is represented.
            #print(a, 'b values', bvals)
            if len(bvals)==0:
                continue
                
            if weight_type == "Linear":
                # linear weighting from the read and write operations. 
                # here it has this many matches a+b with pattern 1 and a+c matches with pattern 2 (maybe the query)
                weighting = ((a+bvals)/n) * ( (a+(dval-bvals))/n )
            elif weight_type == "Cosine":
                # linear weighting from the read and write operations. 
                # here it has this many matches a+b with pattern 1 and a+c matches with pattern 2 (maybe the query)
                weighting = ((a+bvals)/(2*n)) * ( (a+(dval-bvals))/ (2*n) )
            elif weight_type == "Exp":
                # expo weighting from the read and write operations. 
                weighting = np.exp(-exp_beta*(n-(a+bvals))) * np.exp(-exp_beta*(n-(a+(dval-bvals))))
            elif weight_type is None or weight_type == "Binary": 
                weighting = 1
            else: 
                raise NotImplementedError()

            possible_addresses += comb(n-dval,a)*(weighting*comb(dval,bvals)).sum()
        expected_intersect = perc_addresses_w_neurons + np.log(possible_addresses)
        res.append(np.exp(expected_intersect))
    return np.asarray(res)

def fit_beta_regression(n, dvals, res, return_bias=False):
    dvals = np.asarray(dvals)
    xvals = 1-(2*dvals)/n
    res = np.asarray(res)
    zeros_in_res = False
    if res[-1] == 0.0:
        print("res equals 0, problem for the log. removing from the equation here.")
        mask = res!=0.0
        num_zeros = (res==0.0).sum()
        res = res[mask]
        xvals = xvals[mask]
        zeros_in_res = True
    yvals = np.log(np.asarray(res))
    # log linear regression closed form solution. 
    beta = np.cov(xvals, yvals)[0][1] / np.var(xvals)
    b = np.mean(yvals) - beta*np.mean(xvals)
    fit_beta_res = softmax(xvals, beta)
    #print(fit_beta_res)
    #mse between res and beta res: 
    print('MSE:',np.sum((res-fit_beta_res)**2) )
    
    if zeros_in_res: 
        # only true if there were 0 res values. need to append 0s to the end
        fit_beta_res = np.append(fit_beta_res, np.zeros(num_zeros))
    if return_bias: 
        return fit_beta_res, beta, b
    else: 
        return fit_beta_res, beta

def softmax(x, beta):
    assert len(x.shape) <3, 'this softmax can currently only handle vectors'
    x = x * beta
    return np.exp(x)/np.exp(x).sum()