# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, earlystop_criterion, patience=7, verbose=False, delta=0, earlystop_mode = 'minimize', path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.earlystop_criterion = earlystop_criterion
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = earlystop_mode
        if earlystop_mode == 'minimize' : 
            self.best_score = np.Inf
        else : 
            self.best_score = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, loss, auc, model):
        if self.mode == 'minimize' : 
            score = loss
        else :
            score = auc

        if self.mode == 'minimize' : 
            if score > self.best_score - self.delta:
                self.counter += 1
                self.trace_func(f'\t\t\tEarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.trace_func(f'\t\t\t{self.earlystop_criterion} decreased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...\n')
                self.best_score = score
                self.best_loss = loss
                self.best_auc = auc
                self.best_model = model.state_dict()
                self.counter = 0
        else : 
            if score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'\t\t\tEarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.trace_func(f'\t\t\t{self.earlystop_criterion} increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...\n')
                self.best_score = score
                self.best_loss = loss
                self.best_auc = auc
                self.best_model = model.state_dict()
                self.counter = 0

    def save_checkpoint(self):
        torch.save( self.best_model, self.path )