import sys
import os
root_path = os.path.abspath('..')
sys.path.append(root_path)

# Basic Libraries
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss, roc_curve, auc
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt

# Source Codes
from src.seed import seed_everything
from src.logger import print_logger
from src.model_ResNet import ResNeXt50

def func_nll(y_true, y_pred):
    nll = round(log_loss(y_true, y_pred, eps=1e-15, normalize=True),2)
    return nll

def find_youden_index(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def func_specificity(y_true, y_pred):
    cm_generator = ConfusionMatrix(task = 'binary', num_classes=2)
    confusionmatrix = cm_generator(preds = torch.tensor(y_pred), target = torch.tensor(y_true))
    specificity = round(int(confusionmatrix[0,0])/(int(confusionmatrix[0,0])+int(confusionmatrix[0,1])), 2)
    return specificity

class score_loader : 
    def __init__(self, df : pd.DataFrame, level) : 
        self.score_df = df
        self.level = level
        self.total_labels = []
        self.total_preds = []
        self.total_dfs = []
        self.total_tprs = []

    # 95% CI with bootstrap
    def bootstrap_score(self,y_true, y_pred, n_bootstraps=1000, score_type="auc", seed=2023):
        self.score_type = score_type
        yonden_cut = find_youden_index(y_true, y_pred)
        if score_type == "auc" :
            score_func = roc_auc_score
        elif score_type == 'sensitivity':
            score_func = recall_score
            y_pred = y_pred >= 0.5
        elif score_type == 'specificity': 
            score_func = func_specificity
            y_pred = y_pred >= 0.5
        elif score_type == "accuracy" : 
            score_func = accuracy_score
            y_pred = y_pred >= 0.5
        elif score_type == "nll" : 
            score_func = func_nll
            y_pred = y_pred >= 0.5
            nll = []
        elif score_type == "brier" : 
            score_func = brier_score_loss
            y_pred = y_pred >= 0.5 

        np.random.seed(seed)
        bootstrapped_scores = []
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = np.random.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = score_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # 95% CI
        avg_score = np.mean(sorted_scores)
        ci_low = sorted_scores[int(0.025 * len(sorted_scores))]
        ci_high = sorted_scores[int(0.975 * len(sorted_scores))]

        return avg_score, ci_low, ci_high
        
    def load_fold_result(self, fold_num : int, quantile : float) : 
        def get_score(y_true, y_pred) : 
            mean_fpr = np.linspace(0, 1, 100)
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            self.total_tprs.append(interp_tpr)
            auc_score = roc_auc_score(y_true, y_pred)
            df_test = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
            df_test['youden_j'] = df_test['tpr'] - df_test['fpr']
            df_test['youden_j'] = df_test['youden_j'].abs()
            return auc_score, df_test

        score_df = self.score_df.query('outer_fold == @fold_num')
        if self.level == 'patient_level' : 
            patient_level_df = score_df.groupby(['patient_id'])[[f'quantile_{quantile:.2f}', 'severity']].mean().reset_index()
            labels = patient_level_df['severity'].values
            preds = patient_level_df[f'quantile_{quantile:.2f}'].values
        else : 
            labels = score_df['severity'].values
            preds = score_df[f'quantile_{quantile:.2f}'].values
        auc_score, df_test = get_score(labels, preds)

        self.total_labels.extend(labels)
        self.total_preds.extend(preds)
        self.total_dfs.append(df_test)

    def load_all_result(self, quantile) : 
        for fold_num in range(10): #################################### 여기 10으로 바꿔야함
            self.load_fold_result(fold_num, quantile)
            
        self.mean_tpr = np.mean(self.total_tprs, axis=0)
        self.mean_tpr[-1] = 1.0
        mean_auc = auc(np.linspace(0, 1, 100), self.mean_tpr)
        std_tpr = np.std(self.total_tprs, axis=0)
        self.tprs_upper = np.minimum(self.mean_tpr + std_tpr, 1)
        self.tprs_lower = np.maximum(self.mean_tpr - std_tpr, 0)
        

    def load_total_result(self, score_type = "auc") : 
        self.avg_score, self.ci_low, self.ci_high = self.bootstrap_score(np.array(self.total_labels), np.array(self.total_preds), score_type = score_type)
        print(f"{self.score_type if self.score_type != 'auc' else 'AUC'} {self.avg_score :.3f} ({self.ci_low:.3f}-{self.ci_high:.3f})", end = '\n ')

def load_score_df(save_path : Path) : 

    total_score_dfs = []
    for outer_fold_num in range(10) : #########################여기 10으로 바꿔야함
        for inner_fold_num in range(5) : 
            total_score_df_path = save_path.joinpath(f"fold_{outer_fold_num}", f"fold_{inner_fold_num}", "PEPPR_inference_result.csv")
            total_score_df = pd.read_csv(total_score_df_path)
            total_score_df['image_name'] = total_score_df['image_path'].apply(lambda x : Path(x).stem)
            total_score_df['patient_id'] = total_score_df['image_name'].apply(lambda x : x[:-2])
            total_score_df['outer_fold'] = outer_fold_num
            total_score_df['inner_fold'] = inner_fold_num
            total_score_dfs.append(total_score_df)

    total_score_df = pd.concat(total_score_dfs).reset_index(drop=True)
    total_score_df['severity'] = total_score_df['image_name'].map(patient_id_severity_dict)
    return total_score_df

def load_ensemble_scores(score_df : pd.DataFrame, score_metric : str) : 
    assert score_metric in ['auc', 'accuracy', 'sensitivity', 'specificity']

    ensemble_scores = []
    for quantile in np.arange(0, 1.01, 0.05) : 
        ensemble_score_loader = score_loader(score_df, 'patient_level')
        ensemble_score_loader.load_all_result(quantile)
        ensemble_score_loader.load_total_result(score_metric)
        avg_score, ci_low, ci_high = ensemble_score_loader.avg_score, ensemble_score_loader.ci_low, ensemble_score_loader.ci_high
        ensemble_scores.append([quantile, avg_score, ci_low, ci_high])
    return pd.DataFrame(ensemble_scores, columns = ['quantile', 'avg_score', 'ci_low', 'ci_high'])


df_path = Path('total_information.csv')
df = pd.read_csv(df_path)
df['image_stem'] = df['image_path'].apply(lambda x : Path(x).stem)
df['severity'] = df['data_class'].apply(lambda x : 1 if x == 'ASD' else 0)


patient_id_severity_dict = dict(zip(df['image_stem'], df['severity']))

total_scorecam_path = Path('ASD_classification_Total/PEPPR_ScoreCAM')
ados_scorecam_path = Path('ASD_classification_ADOS/PEPPR_ScoreCAM')

total_score_df = load_score_df(total_scorecam_path)
ados_score_df = load_score_df(ados_scorecam_path)

for score_metric in ['auc', 
                        # 'accuracy', 'sensitivity', 'specificity'
                    ] : 
    fig = plt.figure(figsize = (15, 6))
    
    total_ensemble_scores = load_ensemble_scores(total_score_df, score_metric)
    ados_ensemble_scores = load_ensemble_scores(ados_score_df, score_metric)

    quantile_arange = np.arange(0, 1.01, 0.05)

    plt.plot(quantile_arange, total_ensemble_scores['avg_score'].values, color='blue', label = 'ASD (DSM-5 only)')
    plt.fill_between(quantile_arange, total_ensemble_scores['ci_low'].values, total_ensemble_scores['ci_high'].values, color='blue', alpha=0.2)
    plt.scatter(quantile_arange, total_ensemble_scores['avg_score'].values, color='blue', alpha=0.5, s = 10)

    plt.plot(quantile_arange, ados_ensemble_scores['avg_score'].values, color='red', label = 'ASD (DSM-5 and ADOS-2')
    plt.fill_between(quantile_arange, ados_ensemble_scores['ci_low'].values, ados_ensemble_scores['ci_high'].values, color='red', alpha=0.2)
    plt.scatter(quantile_arange, ados_ensemble_scores['avg_score'].values, color='red', alpha=0.5, s = 10)

    plt.xlabel('Fraction of erased image area', fontsize = 15)
    plt.ylabel('AUROC with masked image', fontsize = 15)
    plt.legend(loc = 'lower left', bbox_to_anchor=(0, 1.01), ncol=2, fontsize = 15, frameon=False)
    plt.ylim(0.5, 1.01)
    plt.axhline(0.9, color='black', linestyle='--', alpha=1)

    image_save_path = Path(f'plots/ASD_CLF_TOTAL/PEPPR_ScoreCAM_{score_metric.upper()}_Image.pdf')
    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(image_save_path, dpi=300,  bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
    plt.close()
    break