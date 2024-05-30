"""HMRN individual (I-T or T-I) evaluation"""

from __future__ import print_function
import os
import sys
import time

import torch
import numpy as np

from data import get_test_loader
from model import HMRN
from collections import OrderedDict
import opts
from vg import vg
from vocab import Vocabulary, deserialize_vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    with torch.no_grad():
        for i, (images, captions, captions_msks, lengths, ids) in enumerate(data_loader):
            max_n_word = max(max_n_word, max(lengths))

        for i, (images, captions, captions_msks, lengths, ids) in enumerate(data_loader):
            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, captions_msks)       
            bsize, max_turns = cap_len.size()
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((len(data_loader.dataset), max_turns, cap_emb.size(2)))
                cap_lens = [0] * (len(data_loader.dataset))

            # cache embeddings
            ids = list(ids)
            img_embs[ids] = img_emb.data.cpu().numpy().copy()                                       
            cap_embs[ids, :, :] = cap_emb.data.cpu().numpy().copy()

            for j, nid in enumerate(ids):
                cap_lens[nid] = cap_len[j]                                                           
            new_cap_lens = torch.zeros(len(cap_lens), max_turns, dtype=torch.int64)

            for i in range(len(cap_lens)):
                end = max_turns                                                                     
                new_cap_lens[i, :end] = cap_lens[i]                                                 

            del images, captions
    return img_embs, cap_embs, new_cap_lens                                                      


def evalrank(model_path, split='test'):
    """
    Evaluate a trained model. 
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    # opt = opts.parse_opt()
    save_epoch = checkpoint['epoch']
    print(opt)

    # load dataset
    if 'val' in split:
        db = vg(opt, 'val')
    elif 'test' in split:
        db = vg(opt, 'test')

    # construct model
    model = HMRN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(db, opt.workers, opt.pin_memory)
    print("=> loaded checkpoint_epoch {}".format(save_epoch))

    print('Computing results...')
    with torch.no_grad():  
        img_embs, cap_embs, cap_lens = encode_data(model, data_loader)

        # evaluate the retrieval performance of each round
        for i in range(opt.max_turns):
            print('Images: %d, Retrieval times: %d, %d queries for each retrieval' %
                (img_embs.shape[0]-96, cap_embs.shape[0]-96, i+1))

            # record computation time of validation
            start = time.time()
            sims = shard_attn_scores_test(model, img_embs, cap_embs[:,:i+1,:], cap_lens[:, :i+1], opt, shared_size=700, current_turn=i+1)
            end = time.time()
            print("calculate similarity time:", end-start)

            # save similarity file
            if opt.cross_attention_direction=='I-T':
                np.savez("./results/I2T/sim_{}_round_I2T".format(i+1), sims)
            elif opt.cross_attention_direction=='T-I':
                np.savez("./results/T2I/sim_{}_round_T2I".format(i+1), sims)

            # image retrieval
            ri, _ = eval_test(img_embs, cap_embs[:,:i+1,:], cap_lens[:, :i+1], sims, return_ranks=True)
            print("%d round " % (i+1), end='')
            print("image retrieval results: R1: %.1f, R5: %.1f, R10: %.1f, MR: %.1f" % ri)


def shard_attn_scores_val(model, img_embs, cap_embs, cap_lens, opt, shared_size=200):
    n_im_shard = len(img_embs) // shared_size
    n_cap_shard = len(cap_embs) // shared_size

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shared_size * i, min(shared_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shared_size * j, min(shared_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim_lm, sim_gm, sim_vr = model.forward_sim(im, ca, l)
                # select local-level similarity for current round
                if sim_lm.dim()==3:
                    sim_lm = sim_lm[:, :, -1]
                else:
                    pass
                sim_current_turn = opt.alpha * sim_lm  + opt.beta * sim_vr + (1 - opt.alpha - opt.beta) * sim_gm             

            sims[im_start:im_end, ca_start:ca_end] = sim_current_turn.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def shard_attn_scores_test(model, img_embs, cap_embs, cap_lens, opt, shared_size=200, current_turn = 1):
    n_im_shard = len(img_embs) // shared_size
    n_cap_shard = len(cap_embs) // shared_size

    gcn_turns_sum = 0
    for i in range(1, current_turn + 1):
        gcn_turns_sum += i
    gcn_turns_weight = gcn_turns_sum / (1+2+3+4+5+6+7+8+9+10)
    gcn_turns_weight = round(gcn_turns_weight, 2)

    sims = np.zeros((9800, 9800))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shared_size * i, min(shared_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
                ca_start, ca_end = shared_size * j, min(shared_size * (j + 1), len(cap_embs))

                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim_lm, sim_gm, sim_vr = model.forward_sim(im, ca, l)
                # select local-level similarity for current round
                if sim_lm.dim()==3:
                    sim_lm = sim_lm[:, :, -1]
                else:
                    pass
                sim_current_turn = opt.alpha * sim_lm + opt.beta * gcn_turns_weight * sim_vr +  (1 - opt.alpha - opt.beta) *sim_gm   
                sims[im_start:im_end, ca_start:ca_end] = sim_current_turn.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def eval_val(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = captions.shape[0]                                                          
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

def eval_test(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = captions.shape[0] - 96                                                     
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

if __name__ == '__main__':
    evalrank("./runs/vg/checkpoint/model_best.pth.tar", split="test")
