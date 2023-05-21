import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score




class Trainer : 
    """
    Module for training
    """

    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, logger, device, earlystop, initial_learning_rate) : 
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        self.earlystop = earlystop
        self.train_losses = []
        self.valid_losses = []
        self.current_learning_rate = initial_learning_rate
        self.lr_steps = []

    def train(self, dataloader, phase) : 
        self.model.train()
        self.epoch_loss = 0
        self.epoch_outputs = []
        self.epoch_labels = []
        for X, label, _ in dataloader : 
            X = X.type(torch.FloatTensor).to(self.device, non_blocking = True)
            label = label.flatten().long().to(self.device , non_blocking= True)
            output = self.model(X)
            loss = self.criterion(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epoch_loss += loss.item() 
            self.epoch_outputs.extend(softmax(output)[:,1].detach().cpu().numpy())
            # self.epoch_outputs.extend(softmax(output).detach().cpu().numpy())            
            self.epoch_labels.extend(label.detach().cpu().numpy())
        self.epoch_outputs = np.array(self.epoch_outputs)
        self.epoch_labels = np.array(self.epoch_labels)
        self.calculate_scores()
        self.logger(f'\t\t\t{phase} LOSS : [{round(self.epoch_loss, 5):.4f}]    ACC : [{self.epoch_acc:.4f}]    ROC : [{self.epoch_auc:.4f}]')
        self.train_losses.append(self.epoch_loss)
    
    def valid(self, dataloader, phase) : 
        self.model.eval()
        self.epoch_loss = 0
        self.epoch_outputs = []
        self.epoch_labels = []
        with torch.no_grad() :
            for X, label, _ in dataloader : 
                X = X.type(torch.FloatTensor).to(self.device, non_blocking = True)
                label = label.flatten().long().to(self.device , non_blocking= True)
                output = self.model(X)
                loss = self.criterion(output, label)
                self.epoch_loss += loss.item() 
                self.epoch_outputs.extend(softmax(output)[:,1].detach().cpu().numpy()) 
                self.epoch_labels.extend(label.detach().cpu().numpy())
        self.epoch_outputs = np.array(self.epoch_outputs)
        self.epoch_labels = np.array(self.epoch_labels)
        self.calculate_scores()
        self.logger(f'\t\t\t{phase} LOSS : [{round(self.epoch_loss, 5):.4f}]    ACC : [{self.epoch_acc:.4f}]    ROC : [{self.epoch_auc:.4f}]')

        self.scheduler.step(self.epoch_loss)       
        self.earlystop(self.epoch_loss, self.epoch_auc, self.model)
        self.valid_losses.append(self.epoch_loss)
        if self.optimizer.param_groups[0]['lr'] < self.current_learning_rate : 
            self.current_learning_rate = self.optimizer.param_groups[0]['lr']
            self.lr_steps.append(len(self.valid_losses))

    def test(self, dataloader, phase) : 
        self.model.eval()
        self.epoch_loss = 0
        self.epoch_outputs = []
        self.epoch_labels = []
        with torch.no_grad() :
            for X, label, _ in dataloader : 
                X = X.type(torch.FloatTensor).to(self.device, non_blocking = True)
                label = label.flatten().long().to(self.device , non_blocking= True)
                output = self.model(X)
                loss = self.criterion(output, label)
                self.epoch_loss += loss.item() 
                self.epoch_outputs.extend(softmax(output)[:,1].detach().cpu().numpy())    
                self.epoch_labels.extend(label.detach().cpu().numpy())
        self.epoch_outputs = np.array(self.epoch_outputs)
        self.epoch_labels = np.array(self.epoch_labels)
        self.calculate_scores()
        self.logger(f'\t\t\t{phase}  LOSS : [{round(self.epoch_loss, 5):.4f}]    ACC : [{self.epoch_acc:.4f}]    ROC : [{self.epoch_auc:.4f}]')
        
    def calculate_scores(self) : 
        ###binary
        hard_prediction = [1 if x  >= 0.5 else 0 for x in self.epoch_outputs]
        self.epoch_acc = sum(hard_prediction == self.epoch_labels)/ len(self.epoch_outputs)
        try    : 
            self.epoch_auc = roc_auc_score(self.epoch_labels, self.epoch_outputs)
        except : 
            self.epoch_auc = 0.5

    def fit(self, num_epochs, ) : 
        for num_epoch in range(num_epochs) : 
            self.logger(f'EPOCH : {num_epoch} with LR : [{self.optimizer.param_groups[0]["lr"]}]')
            phase = 'TRAIN'
            self.train(self.dataloaders.get(phase), phase)
            # phase = 'TEST'
            # self.test(self.dataloaders.get(phase), phase)
            phase = 'VALID'
            self.valid(self.dataloaders.get(phase), phase)
            if self.earlystop.early_stop : 
                self.logger(f'==Early Stopped at EPOCH : [{num_epoch}]==\n')
                self.earlystop.save_checkpoint()
                break

        # self.earlystop.load_checkpoint()
        self.epochs = [x for  x in range(num_epoch + 1)]

        path = self.earlystop.path

        best_epoch = len(self.epochs) - self.earlystop.patience
        best_auc = self.earlystop.best_auc
        self.logger(f'==Best AUC : [{best_auc}] at EPOCH : [{best_epoch}] with LR [{self.current_learning_rate}]==\n')