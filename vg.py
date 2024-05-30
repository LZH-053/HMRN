#!/usr/bin/env python

import os, sys, cv2
import json, math
import cairo, pickle
import random, h5py
import numpy as np
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import xml.etree.ElementTree as ET


class vg(object):
    def __init__(self, opt, split=None):
        self.cfg = opt
        self.name = 'vg'
        self.split = split
        self.root_dir  = osp.join(opt.data_path, 'vg')
        self.cache_dir = osp.abspath(osp.join(opt.data_path, 'caches'))  

        with open(osp.join(self.cache_dir, 'vg_vocab_14284.pkl'), 'rb') as fid:
            self.lang_vocab = pickle.load(fid)

        self.load_classes()
        self.load_attributes()
        self.load_relations()
        scenedb = self.load_scenedb()
        scenedb = self.filter_scenedb(scenedb)  
        if self.split is not None:
            split_inds = self.load_split(scenedb, self.split)  
            current_split = {}
            for x in split_inds:
                assert(x == scenedb[x]['image_index']) 
                current_split[x] = scenedb[x]  
            scenedb = current_split  
        self.scenedb = [scenedb[x] for x in sorted(list(scenedb.keys()))]  

    def filter_duplicate_regions(self, regions):
        captions = {}
        filtered_regions = []
        for x in regions:
            c = x['phrase'].lower().encode('utf-8').decode('utf-8')
            if captions.get(c, None) is not None:
                # ignore if the exact caption has appeared before
                continue
            captions[c] = 1
            filtered_regions.append(x)
        return filtered_regions

    def filter_scenedb(self, scenedb):
        filtered_scenedb = {}
        num = len(scenedb)
        for k, scene in scenedb.items():
            if len(scene['regions']) >= self.cfg.max_turns and len(scene['objects']) > 0:  
                # only consider images with at least #max_turns region annotations
                filtered_scenedb[k] = scene
        num_after = len(filtered_scenedb)
        print('Filtered {} scenedb entries: {} -> {} '.format(num - num_after, num, num_after))
        return filtered_scenedb
        
    def load_classes(self):
        self.classes = {0: '__background__'} # each entry contains a list of names, except the '__background__' one
        self.class_to_ind = {}
        self.class_to_ind['__background__'] = 0
        with open(osp.join(self.cache_dir, 'vg_objects_vocab_1600.txt')) as f:
            count = 1
            for obj_alias in f.readlines():
                names = [x.lower().strip() for x in obj_alias.split(',')]
                self.classes[count] = names
                for x in names:
                    self.class_to_ind[x] = count
                count += 1
    
    def load_attributes(self):
        self.attributes = {0: '__no_attribute__'} # each entry contains a list of names, except the '__no_attribute__' one
        self.attribute_to_ind = {}
        self.attribute_to_ind['__no_attribute__'] = 0
        with open(osp.join(self.cache_dir, 'vg_attributes_vocab_1000.txt')) as f:
            count = 1
            for att in f.readlines():
                names = [x.lower().strip() for x in att.split(',')]
                self.attributes[count] = names
                for x in names:
                    self.attribute_to_ind[x] = count
                count += 1 

    def load_relations(self):
        self.relations = {0: '__no_relation__'} # each entry contains a list of names, except the '__no_relation__' one
        self.relation_to_ind = {}
        self.relation_to_ind['__no_relation__'] = 0
        with open(osp.join(self.cache_dir, 'vg_relations_vocab_500.txt')) as f:
            count = 1
            for rel in f.readlines():
                names = [x.lower().strip() for x in rel.split(',')]
                self.relations[count] = names
                for x in names:
                    self.relation_to_ind[x] = count
                count += 1  

    def load_scenedb(self):
        cache_file = osp.join(self.cache_dir, 'vg_scenedb.pkl')
        scenedb = {}
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                data_ = pickle.load(fid)
            for k, v in data_.items():
                scenedb[int(k)] = v
            print('scenedb loaded from {}'.format(cache_file))
        else:
            sg_xml_paths = sorted(glob("%s/sg_xmls/*.xml"%self.root_dir))
            for x in sg_xml_paths:
                image_index = int(osp.splitext(osp.basename(x))[0])
                sg = self.load_sg_annotation(image_index)
                scenedb[image_index] = sg
            # sample negations
            # experiments on negation, can ignore
            # keep it here for now in case removing it will mess up other parts of the codes (e.g. dataloader)
            if self.cfg.negation > 0:
                scenedb = self.sample_negative_objects(scenedb, self.cfg.max_turns//2+1)
            with open(cache_file, 'wb') as fid:
                pickle.dump(scenedb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote scenedb to {}'.format(cache_file))
        return scenedb

    def sample_negative_objects(self, scenedb, K):
        tmpdb = [deepcopy(scenedb[x]) for x in sorted(list(scenedb.keys()))]
        for k, v in scenedb.items():
            negative_objects = {}
            positive_names = [o['name'] for _, o in v['objects'].items()]
            while len(negative_objects) < K:
                rand_id = np.random.permutation(range(len(tmpdb)))[0]
                negative_scene = tmpdb[rand_id]
                for obj_id, obj in negative_scene['objects'].items():
                    if not (obj['name'] in positive_names):
                        positive_names.append(obj['name'])
                        negative_objects[obj_id] = deepcopy(obj)
            # print('sample negative:', k)
            v['negative_objects'] = negative_objects
        return scenedb
                
    def load_sg_annotation(self, image_index):
        xml_path = osp.join(self.root_dir, 'sg_xmls', '%d.xml'%image_index)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag == 'image_id':
                assert(int(elem.text) == image_index)
            elif elem.tag == 'width':
                width = int(elem.text)
            elif elem.tag == 'height':
                height = int(elem.text)
        xml_objs = tree.findall('object')
        xml_rels = tree.findall('relation')

        obj_dict = {}
        for i in range(len(xml_objs)):
            xml_obj = xml_objs[i]
            obj_name = xml_obj.find('name').text.lower().strip()
            if obj_name in self.class_to_ind:
                bbox = xml_obj.find('bndbox')
                x1 = max(0,float(bbox.find('xmin').text))
                y1 = max(0,float(bbox.find('ymin').text))
                x2 = min(width-1,float(bbox.find('xmax').text))
                y2 = min(height-1,float(bbox.find('ymax').text))
                if x1 > x2 or y1 > y2:
                    continue
                obj_idx = int(xml_obj.find('object_id').text)
                xml_atts = xml_obj.findall('attribute')
                obj_atts = []
                for att in xml_atts:
                    att = att.text.lower().strip()
                    if att in self.attribute_to_ind:
                        obj_atts.append(att)
                obj_dict[obj_idx] = {
                    'name': str(obj_name),
                    'idx':  obj_idx,
                    'xyxy': [x1, y1, x2, y2],
                    'atts': obj_atts,
                    'regions': []
                }

        rel_dict = {}
        for i in range(len(xml_rels)):
            xml_rel = xml_rels[i]
            pred = xml_rel.find('predicate').text
            if pred: 
                pred = pred.lower().strip()
                if pred in self.relation_to_ind:
                    rel_id = int(xml_rel.find('relationship_id').text)
                    subject_id = int(xml_rel.find('subject_id').text)
                    object_id  = int(xml_rel.find('object_id').text)
                    if (subject_id in obj_dict) and (object_id in obj_dict):
                        rel_dict[rel_id] = {
                            'subject_id': subject_id,
                            'object_id': object_id,
                            'predicate': pred,
                            'regions': []
                        }

        rg_path = osp.join(self.root_dir, 'rg_jsons', '%d.json'%image_index)
        with open(rg_path, 'r') as fid:
            region_data = json.load(fid)
        regions = {}
        for r in self.filter_duplicate_regions(region_data['regions']):
            regions[int(r['region_id'])] = r
        meta_regions = {}
        for k, current_region in regions.items():
            x1 = max(current_region['x'], 0)
            y1 = max(current_region['y'], 0)
            x2 = min(width-1, x1 + current_region['width'])
            y2 = min(height-1, y1 + current_region['height'])
            meta_regions[k] = {
                'index': k, 
                'xyxy': [x1, y1, x2, y2],
                'caption': current_region['phrase'].lower().encode('utf-8').decode('utf-8')
            }

            # Associate regions with objects and relations
            for o in current_region['objects']:
                current_region_obj_idx = o['object_id']
                if current_region_obj_idx in obj_dict:
                    obj_dict[current_region_obj_idx]['regions'].append(k)
            for r in current_region['relationships']:
                current_region_rel_idx = r['relationship_id']
                if current_region_rel_idx in rel_dict:
                    rel_dict[current_region_rel_idx]['regions'].append(k)
            
        scene = {
            'image_index': image_index, 
            'width': width, 
            'height': height, 
            'objects': obj_dict, 
            'relations': rel_dict,
            'regions': meta_regions
        }
        return scene
  
    def load_split(self, scenedb, split):
        cache_file = osp.join(self.cache_dir, 'vg_%s.txt'%split)
        if osp.exists(cache_file):
            split_img_inds = list(np.loadtxt(cache_file, dtype=np.int32))
        else:
            # As far as I remember the 'raw_test.txt' file contains images from the test set of the work:
            # "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering",
            # which were not used in Faster RCNN training
            all_image_inds = set([k for k, v in scenedb.items()])
            test_inds = set(list(np.loadtxt(osp.join(self.cache_dir, 'raw_test.txt'), dtype=np.int32)))
            rest_inds = list(all_image_inds.difference(test_inds))
            test_inds = list(all_image_inds.intersection(test_inds))
            rand_ref_inds = np.random.permutation(range(len(rest_inds)))
            train_ref_inds = rand_ref_inds[:-5000]
            val_ref_inds   = rand_ref_inds[-5000:]
            image_indices = {}
            image_indices['train'] = sorted([rest_inds[x] for x in train_ref_inds])
            image_indices['val']   = sorted([rest_inds[x] for x in val_ref_inds])
            image_indices['test']  = sorted(test_inds)
            for x in ['train', 'val', 'test']:
                path = osp.join(self.cache_dir, 'vg_%s.txt'%x)
                print(x, len(image_indices[x]))
                np.savetxt(path, sorted(image_indices[x]), fmt='%d')
            split_img_inds = image_indices[split]    
        return split_img_inds

    def region_path_from_index(self, index):
        return osp.join(self.root_dir, 'region_36_final', str(index).zfill(12) + '.npy')
