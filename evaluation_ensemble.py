"""HMRN ensemble evaluation"""

from __future__ import print_function
import numpy as np
import opts

def ensemble_eval(sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = 9800                                                      
    ranks = np.zeros(npts)                                                              
    top1 = np.zeros(npts)                                                                                                                       

    # --> (N(caption), N(image))
    sims = sims.T

    for index in range(npts):                                                          
        inds = np.argsort(sims[index])[::-1]                                           
        # score
        rank = np.where(inds == index)[0][0]                                           
        ranks[index] = rank                                                             
        top1[index] = inds[0]                                                                                                              

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, meanr)
    
def eval_all():
    '''
    evaluate the retrieval results for all rounds
    '''
    opt = opts.parse_opt()
    for i in range(opt.max_turns):
        # select N-th round similarities for evaluation
        sim_I2T = np.load("./results/I2T/sim_{}_round_I2T.npz".format(i+1))['arr_0']
        sim_T2I = np.load("./results/T2I/sim_{}_round_T2I.npz".format(i+1))['arr_0']
        sim_both = (sim_T2I + sim_I2T) / 2
        ri, _ = ensemble_eval(sim_both, return_ranks=True)
        print("%d round " % (i+1), end='')
        print("image retrieval results: R1: %.1f, R5: %.1f, R10: %.1f, MR: %.1f" % ri)


if __name__ == '__main__':
    eval_all()
