"""Data provider"""
#!/usr/bin/env python

import os, sys, cv2, json
import random, pickle, math
import numpy as np
import os.path as osp
from PIL import Image
from time import time
from copy import deepcopy
from glob import glob
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class region_loader(Dataset):
    """
    Load precomputed captions and image features
    """
    def __init__(self, imdb):  # imdb, train_db/val_db/test_db
        self.cfg = imdb.cfg
        self.db = imdb  
        
    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, scene_index):
        scene = self.db.scenedb[scene_index]  # scene index
        image_index = scene['image_index']    # image index

        # load region feature
        region_path = self.db.region_path_from_index(image_index)  
        with open(region_path, 'rb') as fid:
            regions = pickle.load(fid, encoding='latin1')  
        region_feats = torch.from_numpy(regions['region_feats']).float() 

        # load region caption
        all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]  
        all_captions = [x['caption'] for x in all_meta_regions]  

        if self.db.split in ['val', 'test']:
            captions = all_captions[:self.cfg.max_turns]  # val set and test set are kept in a fixed order, no need to shuffle the captions
        else:
            num_captions = len(all_captions)
            caption_inds = np.random.permutation(range(num_captions))  # shuffle the captions for training set
            captions = [all_captions[x] for x in caption_inds[:self.cfg.max_turns]]  # pick up the first ten captions after shuffling
        
        sent_inds = []
        for i in range(self.cfg.max_turns):
            tokens = [w for w in word_tokenize(captions[i])]  
            word_inds = [self.db.lang_vocab(w) for w in tokens]  
            word_inds.append(self.cfg.EOS_idx)  
            sent_inds.append(torch.Tensor(word_inds))  
        sent_inds = tuple(sent_inds)

        return region_feats, sent_inds, scene_index, image_index


def region_collate_fn(data):
    region_feats, sent_inds, scene_indices, image_indices = zip(*data)
    bsize=len(region_feats)
    # regions
    max_length = 36  # the number of selected regions in each image
    new_region_feats  = torch.zeros(len(region_feats), max_length, region_feats[0].size(-1)).float()  

    for i in range(len(region_feats)):  
        end = region_feats[i].size(0)  
        new_region_feats[i, :end] = region_feats[i]  

    # captions
    lengths = [len(sent_inds[i][j]) for i in range(len(sent_inds)) for j in range(len(sent_inds[0]))]
    max_length = max(lengths)  
    new_sent_inds = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()  
    new_sent_msks = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()  
    for i in range(len(sent_inds)):
        for j in range(len(sent_inds[0])):
            end = len(sent_inds[i][j])  
            new_sent_inds[i, j, :end] = sent_inds[i][j]  
            new_sent_msks[i, j, :end] = 1  

    return new_region_feats, new_sent_inds, new_sent_msks, lengths, scene_indices  

def get_precomp_loader(db, batch_size=100, shuffle=True, num_workers=4, pin_memory=False):

    dset = region_loader(db)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              collate_fn=region_collate_fn)
    return data_loader


def get_loaders(train_db, val_db, batch_size, workers, pin_memory):
 
    # get the train_loader
    train_loader = get_precomp_loader(train_db, batch_size, True, workers, pin_memory)
    # get the val_loader
    val_loader = get_precomp_loader(val_db, 100, False, workers, pin_memory)

    return train_loader, val_loader


def get_test_loader(test_db, workers, pin_memory):

    # get the test_loader
    test_loader = get_precomp_loader(test_db, 100, False, workers, pin_memory)

    return test_loader
