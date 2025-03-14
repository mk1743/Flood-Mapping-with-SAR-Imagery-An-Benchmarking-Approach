#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Source: https://github.com/getch-geohum/UDA-SOLO/blob/master/scripts/ot_runner.py

import torch
import numpy as np
import ot


def compute_lamda_stepwise(n_epoch, dstep, d_epoch, len_loader): # (11) from paper
    '''
    n_epoch: total number of training epochs
    dstep: trainning step within a single epoch (iteration)
    d_epoch: specific epoch within training epochs (epoch)
    len_loader: length of Data loader
    '''
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    return alpha


class FeatureAlignment: # on batch level
    def __init__(self, reduction='sum', alpha=1):
        '''Computes the domain alignment at the feature space'''
        self.alpha = alpha
        self.reduction = reduction
    
    '''
    x1, x2 predictions 
    '''

    def compute(self, x1, x2,reduce_sdim=True, mapping=True): # unsorted values 
        #assert x1.shape == x2.shape, f'the source and target shapes {x1.shape} and {x2.shape} respectivel are not teh same'
        if reduce_sdim: # 2 with, 3 height; b,c,w,h --> 0,1,2,3
            x1 = torch.nanmean(x1, dim=(2,3)) # gives [b, #f] and reduces the spatial dimension
            x2 = torch.nanmean(x2, dim=(2,3))
            assert x1.shape == x2.shape, f'the source and target shapes {x1.shape} and {x2.shape} respectivel are not the same'
        a = x1.shape[0]
        b = x2.shape[0]
        #print('inf_nan ff: ', torch.isnan(x1).any(), torch.isnan(x1).any(), torch.isinf(x1).any(), torch.isinf(x1).any())
        A = torch.from_numpy(ot.unif(a)).cuda()
        B = torch.from_numpy(ot.unif(b)).cuda()
        cost = ot.dist(x1.view(a, -1), x2.view(b, -1), metric='euclidean')
        if not cost.is_cuda:
            cost.cuda()
        gamma = ot.emd(A, B, cost) # coulping matrix [batch_size,batch_size]

        if mapping:
            sorting = torch.argmax(gamma, dim=0).detach().cpu().numpy().tolist() # if we need to sort the source ground truth that are mmaped with target images
            #sorting = torch.argmax(gamma, dim=1).detach().cpu().numpy().tolist() # if we need to sort the target input to label space alignment(unified masks predicted from the mask branch)
        else:
            sorting = None

        loss = self.alpha*gamma*cost
        if self.reduction == 'mean':
            return loss.mean(), sorting
        elif self.reduction == 'sum':
            return loss.sum()/a, sorting
        else:
            raise ValueError(f'The reduction type {self.reduction} not known!')

class LabeAlignment:
    '''
    x1 = model prediction from the target domain
    x2 = sorted mask from the source domain
    '''
    def __init__(self, beta=1, reduction='sum'):
        self.beta = beta
        self.reduction=reduction
        '''Computes label space distance'''
    def compute(self, x1, x2):
        assert x1.shape == x2.shape, f'the source and target shapes {x1.shape} and {x2.shape} respectivel are not the same'
        a = x1.shape[0]
        b = x2.shape[0]
        #print('inf_nan ll: ', torch.isnan(x1).any(), torch.isnan(x1).any(), torch.isinf(x1).any(), torch.isinf(x1).any())
        A = torch.from_numpy(ot.unif(a)).cuda()
        B = torch.from_numpy(ot.unif(b)).cuda()
        cost = ot.dist(x1.view(a, -1), x2.view(b, -1).float().cuda(), metric='euclidean')
    
        if not cost.is_cuda:
            cost.cuda()
        gamma = ot.emd(A, B, cost)
        if not gamma.is_cuda:
            gamma.cuda()
        label_loss = self.beta*gamma*cost
        if self.reduction=='sum':
            return label_loss.sum()/a
        elif self.reduction=='mean':
            return label_loss.mean() #.mean()  # sum can also be considered
        else:
            raise ValueError(f'The reduction type {self.reduction} not known!')


# step wise lamda compute
def compute_alpha(n_epoch, dstep, d_epoch, len_loader):
    '''
    n_epoch: total number of training epochs
    dstep: trainning step within a single epoch
    d_epoch: specific epoch within training epochs
    len_loader: length of Data loader
    '''
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    return alpha

# step wise LR adjuster
def compute_adapt_lr(muno_lr, n_epoch, dstep, d_epoch, len_loader):
    p = float(dstep + d_epoch * len_loader) / n_epoch / len_loader
    step_lr = muno_lr/(1+10*p)**0.75
    return step_lr

def order(x, inds): # sorting function for the label alignment class 
    return [x[ind] for ind in inds]

    
    


































