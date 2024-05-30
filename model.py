"""HMRN model"""

from turtle import forward
import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):
    """
    Build region representations by common-used FC-layer.
    Args: - images: raw regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: final region embeddings, shape:  (batch_size, 36, embed_size).
    """
    def __init__(self, opt, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build caption representations by common-used Bi-GRU.
    Args: - captions: raw caption ids, shape: (batch_size, max_turns, L).
    Returns: - cap_emb: final caption embeddings, shape: (batch_size, max_turns, embed_size).
    """
    def __init__(self, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, captions_msks):
        """Handles variable size captions, output fixed-dimension features"""
        '''
        Args: - captions, shape(batch_size, max_turns, max_length)
              - lengths, shape(batch_size, max_turns)
        '''
        # embed word ids to vectors (hacky clamp)
        bsize, max_turns, nwords = captions.size()
        captions = captions.view((-1, nwords))  
        captions_msks=captions_msks.view(-1, nwords)                                                              

        lengths = torch.sum(captions_msks, -1).clamp(min=1).long()                                                
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)                                      
        _, unsorted_indices = torch.sort(sorted_indices)                                                           

        cap_emb = self.embed(captions[sorted_indices])                                                             
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(cap_emb, sorted_lengths, batch_first=True)                                  

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] + cap_emb[:, :, cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        
        cap_emb = cap_emb[unsorted_indices]                                                                        

        I = lengths.view(-1, 1, 1).cuda()
        I = I.expand(cap_emb.size(0), 1, self.embed_size) - 1
        last_feat = torch.gather(cap_emb, 1, I).squeeze(1)
        last_feat = last_feat.view(bsize, max_turns, self.embed_size)                                              
        lengths = lengths.view(bsize, max_turns)                                                                   
        return last_feat, lengths                                                                                        

class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, embed_size)
          - raw_global: raw image by averaging regions, shape: (batch_size, embed_size)
    Returns: - new_global: final image by self-attention, shape: (batch_size, embed_size).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)                       
        g_emb = self.embedding_global(raw_global)                  

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)     
        common = l_emb.mul(g_emb)                                  
        weights = self.embedding_common(common).squeeze(2)         
        weights = self.softmax(weights)                            

        # compute final image, shape: (batch_size, embed_size)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)     
        new_global = l2norm(new_global, dim=-1)                    

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: caption embeddings, shape: (batch_size, max_turns, embed_size)
          - raw_global: raw text by averaging captions, shape: (batch_size, embed_size)
    Returns: - new_global: final text by self-attention, shape: (batch_size, embed_size).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of captions and raw global text
        l_emb = self.embedding_local(local)                         
        g_emb = self.embedding_global(raw_global)                   

        # compute the normalized weights
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)      
        common = l_emb.mul(g_emb)                                   
        weights = self.embedding_common(common).squeeze(2)          
        weights = self.softmax(weights)                             

        # compute final text, shape: (batch_size, embed_size)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)     
        new_global = l2norm(new_global, dim=-1)                     

        return new_global

class InterCorrelationReasoning(nn.Module):
    """
    Perform the inter-correlation reasoning with a full-connected graph
    Args: - sim_emb: intra-correlation vector, shape: (batch_size, max_turns + 1, embed_size)
    Returns; - sim_icr: inter-correlation reasoned graph nodes, shape: (batch_size, max_turns + 1, embed_size)
    """
    def __init__(self, sim_dim):
        super(InterCorrelationReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)                                             
        sim_key = self.graph_key_w(sim_emb)                                                 
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)    
        sim_icr = torch.bmm(sim_edge, sim_emb)                                              
        sim_icr = self.relu(self.sim_graph_w(sim_icr))
        return sim_icr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by scalar-based matching and vector-based reasoning
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, embed_size)
          - cap_emb: local caption embeddings, shape: (batch_size, max_turns, embed_size)
    Returns:
        - sim_lm: local-level matching similarity for each round (I-T branch or T-I branch), shape: (batch_size, batch_size, max_turns)
        - sim_gm: global-level matching similarity, shape: (batch_size, batch_size)
        - sim_vr: vector-based reasoning similarity, shape: (batch_size, batch_size)
    """
    def __init__(self, opt, embed_size, sim_dim, icr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.opt = opt

        self.v_global_w = VisualSA(embed_size, 0.4, 36)                              
        self.t_global_w = TextSA(embed_size, 0.4)                                    
        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.ICR_module = nn.ModuleList([InterCorrelationReasoning(sim_dim) for i in range(icr_step)])

        self.init_weights()

    def compute_pairwise_similarity(self, src_feats, tgt_feats):
        sim = torch.bmm(tgt_feats, src_feats.transpose(1, 2))                                        
        sim = nn.LeakyReLU(0.1)(sim)  
        return sim  

    def pairwise_similarity_to_attn(self, pairwise_similarities, temperature_lambda):
        attn = temperature_lambda * pairwise_similarities.clamp(min=-1e10)                  
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]                         
        attn = F.softmax(attn, dim=-1)                                                              
        return attn

    def forward(self, img_emb, cap_emb, cap_lens):                                                                                                                  
        bsize, n_regions, embed_size = img_emb.size()
        bsize, max_turns, embed_size = cap_emb.size()

        # compute_batch_mutual_similarity
        # select cross-attention directions, i.e., I-T or T-I
        if self.opt.cross_attention_direction == 'I-T':
            # i2t
            region_feats = img_emb.view(1, bsize, n_regions, embed_size)
            region_feats = region_feats.expand(bsize, bsize, n_regions, embed_size).contiguous()
            region_feats = region_feats.view(bsize, bsize * n_regions, embed_size).contiguous()
            # attended similarity on queries
            sim_AllRound_region = torch.zeros(bsize, bsize, max_turns).cuda()                              
            for i in range(max_turns):
                cap_emb_i = cap_emb[:, :i+1, :]                                                               
                sim_region = self.compute_pairwise_similarity(cap_emb_i, region_feats)                         
                attn_region = self.pairwise_similarity_to_attn(sim_region, self.opt.temperature_lambda_i2t)        
                sim_CurrentRound_region = torch.sum(sim_region * attn_region, dim=-1)                          
                sim_CurrentRound_region = sim_CurrentRound_region.view(bsize, bsize, n_regions)                
                sim_CurrentRound_region = torch.mean(sim_CurrentRound_region, -1)                              
                sim_AllRound_region[:, :, i] = sim_CurrentRound_region   

        elif self.opt.cross_attention_direction == 'T-I':
            # t2i
            query_feats = cap_emb.view(1, bsize, max_turns, embed_size)
            query_feats = query_feats.expand(bsize, bsize, max_turns, embed_size).contiguous()
            query_feats = query_feats.view(bsize, bsize * max_turns, embed_size).contiguous()
            # attended similarity on regions
            sim_query = self.compute_pairwise_similarity(img_emb, query_feats)                          
            attn_query = self.pairwise_similarity_to_attn(sim_query, self.opt.temperature_lambda_t2i)        
            sim_all_query = torch.sum(sim_query * attn_query, dim=-1)                                    
            sim_all_query = sim_all_query.view(bsize, bsize, max_turns)  

        else:
            print('cross attention direction error!')                                

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)                                                               
        img_glo = self.v_global_w(img_emb, img_ave)                                                  

        # get enhanced global captions by self-attention
        cap_ave = torch.mean(cap_emb, 1)                                                             
        cap_glo = self.t_global_w(cap_emb, cap_ave)                                                  

        # compute batch global similarity
        sim_gm = torch.mm(img_glo, cap_glo.t())                                                     

        sim_vr = []
        for i in range(bsize):
            cap_i = cap_emb[i, :, :].unsqueeze(0)                                                       
            cap_i_expand = cap_i.repeat(bsize, 1, 1)                                                   
            cap_glo_i = cap_glo[i, :].unsqueeze(0)                                          

            Context_img = SCAN_attention(cap_i_expand, img_emb, self.opt.temperature_lambda_t2i)                              
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)                                 
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)                                        

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)                                        
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)                                       
 
            # concat global and local region-query intra-correlation vectors
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)                             

            # inter-correlation reasoning
            for module in self.ICR_module:
                sim_emb = module(sim_emb)
            sim_vec = sim_emb[:, 0, :]                                                                   
        
            # compute the final high-level reasoning similarity
            sim_i = self.sigmoid(self.sim_w(sim_vec))                                                 
            sim_vr.append(sim_i)                                                                       

        sim_vr = torch.cat(sim_vr, 1)                                                                  

        # compute final similarity. The N-th round similarity is obtained by averaging N respective similarities.
        if self.opt.cross_attention_direction == 'I-T':
        # similarity calculation for I2T branch
            sim_lm = sim_AllRound_region      
        elif self.opt.cross_attention_direction == 'T-I':                                                  
        # similarity calculation for T2I branch
            sim_lm = sim_all_query                                                                        
            sim_temp = torch.zeros(bsize, bsize, max_turns).cuda()
            for i in range(sim_lm.size(-1)):
                sim_temp[:, :, i] = torch.mean(sim_lm[:, :, 0 : i+1], dim = -1)
            sim_lm = sim_temp       
        else:
            print('cross attention direction error!')  

        return sim_lm, sim_gm, sim_vr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def SCAN_attention(query, context, temperature_lambda_t2i, eps=1e-8):
    """
    query: (n_context, queryL, d), (bsize, current_caption_length, embed_size)
    context: (n_context, sourceL, d), (bsize, 36, embed_size)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)                                

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)                                        
    attn = torch.bmm(context, queryT)                                     

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()                       
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn * temperature_lambda_t2i, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()                      

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)                             
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)                         
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)              
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext

class infoNCELoss(nn.Module):

    def __init__(self, tau=1):
        super(infoNCELoss, self).__init__()
        self.tau = tau

    def forward(self, scores):                                             

        bsize, bsize = scores.size()
        scores = self.tau * scores.clamp(min=-1e10)
        d1 = F.log_softmax(scores, dim=1)                               
        d2 = F.log_softmax(scores, dim=0)                                 

        loss_s = torch.sum(d1.diag())
        loss_im = torch.sum(d2.diag())
        loss_infoNCE = -1 * (loss_s + loss_im) / bsize                     

        return loss_infoNCE

class ComputeFinalLoss(nn.Module):

    def __init__(self, opt, alpha=0.6, beta=0.3):
        super(ComputeFinalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta      
        self.criterion = infoNCELoss(tau=opt.tau)

    def forward(self, sim_lm, sim_gm, sim_vr):                                         

        bsize, bsize, max_turns = sim_lm.size()

        # compute infoNCE loss  
        # calculate local-level matching loss for each round
        loss_lm_infoNCE = [self.criterion(sim_lm[:, :, 0])]   
        if max_turns > 1:                                                  
            for i in range(1, max_turns):
                loss_lm_infoNCE.append(self.criterion(sim_lm[:, :, i]))                            
        loss_lm_infoNCE = torch.stack(loss_lm_infoNCE, -1)                                                                                      
        loss_lm_infoNCE = torch.mean(loss_lm_infoNCE, -1)                                

        # calculate global-level matching loss
        loss_gm_infoNCE = self.criterion(sim_gm)

        # calculate vector-based reasoning loss                                      
        loss_vr_infoNCE = self.criterion(sim_vr)   

        # sum infoNCE loss
        loss_infoNCE =  self.alpha * loss_lm_infoNCE + self.beta * loss_vr_infoNCE + (1 - self.alpha - self.beta) * loss_gm_infoNCE    

        #  FINAL LOSS
        loss_all = loss_infoNCE

        return loss_all


class HMRN(object):
    """
    Hierarchical Matching and Reasoning Network (HMRN)
    """
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt, opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt, opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_enc = EncoderSimilarity(opt, opt.embed_size, opt.sim_dim,
                                        opt.icr_step)
        self.compute_loss = ComputeFinalLoss(opt, opt.alpha, opt.beta)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params    

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, captions_msks):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding                                                                    
        img_embs = self.img_enc(images)                                                              
        cap_embs, lengths = self.txt_enc(captions, captions_msks)                                                                                                              
        return img_embs, cap_embs, lengths                                                              

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sim_lm, sim_gm, sim_vr = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sim_lm, sim_gm, sim_vr                                                        

    def forward_loss(self, sim_lm, sim_gm, sim_vr, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        bsize = sim_lm.size(0) 
        loss = self.compute_loss(sim_lm, sim_gm, sim_vr)
        self.logger.update('infoNCE', loss.item(), bsize)
        return loss

    def train_emb(self, images, captions, captions_msks, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, captions_msks)
        sim_lm, sim_gm, sim_vr = self.forward_sim(img_embs, cap_embs, cap_lens)                                       

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sim_lm, sim_gm, sim_vr)                                                             

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

