from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,log_loss
import pandas as pd
from torch.nn.functional import softmax
from torchmetrics.classification import ConfusionMatrix
import torch
import warnings
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from src.seed import seed_everything

seed_everything(42)


# 오류 경고 무시하기
warnings.filterwarnings(action='ignore')
# 오류 메세지 다시 보이게 하기
# warnings.filterwarnings(action='default')

from src.dataloader_setter import Dataloader_Setter
from src.logger import print_logger
from src.load_model import load_model

#### 변경이 필요한 부분
batch_size = 1
device = 'cuda'
initial_learning_rate = 0.001

####
# objective = sys.argv[1]
log_name = Path('/home/jaehan0605/MAIN_ASDfundus/ResNeXt50_deepensemble_SRS_76')
total_path = Path(log_name)

logger = print_logger(total_path, "Total_Inference.txt")
dataloader = Dataloader_Setter(logger)

df = pd.read_csv('DATA/Processed_dataset/dataset.csv').query('FOLD_0==2')[['image_path', 'data_class', 'patient_id']].reset_index(drop=True)

if __name__ == "__main__" : 

    fold_dataloader = dataloader.load(0, batch_size)
    test_dataloader = fold_dataloader.get("TEST")
    logger(f'TEST Dataloader : {test_dataloader.dataset.__len__()} | {np.bincount(test_dataloader.dataset.labels)}', end = '\n')
    logger(f"FOLD NUM   | AUROC     | Sensitivity  | Speicificty | Accuaracy    | NLL       | Brier     ")
    
    total_scores_image_level = []
    total_scores_patient_level = []

    for fold_num in range(10) :         
        fold_path = total_path.joinpath(f'fold_{fold_num}')
        fold_logger = print_logger(fold_path, "inference_log.txt")
        
        preds_ensemble = [] #List for deep ensemble in retinal fundus image level
        preds_ensemble_patient = [] #List for deep ensemble in patient level

        # Load Trained Model
        model = load_model(model_name='ResNeXt50', pretrained=True, num_classes=2)

        globals()['df_{}'.format(fold_num)] = pd.DataFrame()
        
        for model_num in range(5):
            state_dict = torch.load(fold_path.joinpath(f'Model_{model_num}.pth'))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # inference
            with torch.no_grad():
                preds = []
                labels = []
                patient_id = []

                for X, y, path_stem in test_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    pred = model(X)
                    preds.extend(softmax(pred)[:,1].cpu().numpy())
                    labels.extend(y.cpu().numpy())
                    patient_id.extend(path_stem)
                    
                list_preds = preds
                list_labels = labels
                preds = np.array(preds)
                labels = np.array(labels)

                preds_ensemble.append(list_preds)

                globals()['df_{}'.format(fold_num)]['patient_id'] = patient_id
                globals()['df_{}'.format(fold_num)][f'model_{model_num}'] = list_preds
                globals()['df_{}'.format(fold_num)]['label'] = list_labels                
                globals()['df_{}'.format(fold_num)]['patient_name'] =  globals()['df_{}'.format(fold_num)]['patient_id'].apply(lambda x : int(x.split('_')[1]))
            
                mean_result = globals()['df_{}'.format(fold_num)].groupby(['patient_name'])[f'model_{model_num}'].mean().to_dict()
                globals()['df_{}'.format(fold_num)]['patient_name'] = globals()['df_{}'.format(fold_num)]['patient_id'].apply(lambda x : int(x.split('_')[1]))
                globals()['df_{}'.format(fold_num)][f'patient_level_pred_{model_num}'] = globals()['df_{}'.format(fold_num)]['patient_name'].map(mean_result)
                
                if model_num ==4:
                    globals()['df_{}'.format(fold_num)].drop_duplicates(subset=['patient_name', f'patient_level_pred_{model_num}'], inplace=True)
                    for jhk in range(5):
                        preds_model = globals()['df_{}'.format(fold_num)][f'patient_level_pred_{jhk}']
                        labels_model = globals()['df_{}'.format(fold_num)]['label']
                        np.save(fold_path.joinpath(f'patient_level_preds_singlemodel_{jhk}.npy'), preds_model)
                        np.save(fold_path.joinpath(f'patient_level_labels_singlemodel.npy'), labels_model)
                else: 
                    pass
                
            # calculate scores
            ## scores : [AUC, Accuracy, Precision, Recall, F1, NLL, Brier]
            soft_preds = preds.copy()
            hard_preds = np.where(soft_preds >= 0.5, 1, 0)

            cm_generator = ConfusionMatrix(task = 'binary', num_classes=2)
            confusionmatrix = cm_generator(preds = torch.tensor(hard_preds), target = torch.tensor(labels))

            auc_score   = str(round(roc_auc_score(labels, soft_preds)   * 100, 2))
            sensitivity = str(round(recall_score(labels, hard_preds)    * 100, 2))
            specificity = str(round(int(confusionmatrix[0,0])/(int(confusionmatrix[0,0])+int(confusionmatrix[0,1]))*100, 2))
            accuracy    = str(round(accuracy_score(labels, hard_preds)  * 100, 2))
            nll         = str(round(log_loss(labels, hard_preds, eps=1e-15, normalize=True),2))
            brier       = str(round(brier_score_loss(labels, hard_preds),2))

            fold_logger(f"{str(fold_num).ljust(11, ' ')}| {auc_score.ljust(10, ' ')}| {sensitivity.ljust(10, ' ')}| {specificity.ljust(10, ' ')}| {accuracy.ljust(10, ' ')}| {nll.ljust(10, ' ')}| {brier.ljust(10, ' ')}"  )
            fold_logger(f"\n{confusionmatrix}")

        #######ensemble results at image level
        preds_df_folds = pd.DataFrame(preds_ensemble)
        preds_ensemble_list = []
        
        for col in preds_df_folds.columns:
            p_for_severe = preds_df_folds[col].mean()
            preds_ensemble_list.append(p_for_severe)

        preds_image_level = np.array(preds_ensemble_list)
        labels_image_level = labels

        np.save(fold_path.joinpath('image_level_preds_ensemble.npy'), preds_image_level)
        np.save(fold_path.joinpath('image_level_labels_ensemble.npy'), labels_image_level)

        soft_preds_image_level = preds_image_level.copy()
        hard_preds_image_level = np.where(soft_preds_image_level >= 0.5, 1, 0)

        cm_generator = ConfusionMatrix(task = 'binary', num_classes=2)
        confusionmatrix_image_level = cm_generator(preds = torch.tensor(hard_preds_image_level), target = torch.tensor(labels_image_level))

        auc_score_image_level   = str(round(roc_auc_score(labels_image_level, soft_preds_image_level)* 100, 2))
        sensitivity_image_level = str(round(recall_score(labels_image_level, hard_preds_image_level)    * 100, 2))
        specificity_image_level = str(round(int(confusionmatrix_image_level[0,0])/(int(confusionmatrix_image_level[0,0])+int(confusionmatrix_image_level[0,1]))*100, 2))
        accuracy_image_level    = str(round(accuracy_score(labels_image_level, hard_preds_image_level)  * 100, 2))
        nll_image_level         = str(round(log_loss(labels_image_level, hard_preds_image_level, eps=1e-15, normalize=True),2))
        brier_image_level       = str(round(brier_score_loss(labels_image_level, hard_preds_image_level),2))

        fold_logger(f"'ensemble_image_level'|{auc_score_image_level.ljust(10, ' ')}| {sensitivity_image_level.ljust(10, ' ')}| {specificity_image_level.ljust(10, ' ')}| {accuracy_image_level.ljust(10, ' ')}| {nll_image_level.ljust(10, ' ')}| {brier_image_level.ljust(10, ' ')}"  )
        fold_logger(f"\n{confusionmatrix_image_level}")
        total_scores_image_level.append([fold_num, auc_score_image_level, sensitivity_image_level, specificity_image_level, accuracy_image_level, nll_image_level, brier_image_level])         
        
        #######ensemble results at patient level
        df['preds'] = preds_image_level
        df['labels'] = df['data_class'].apply(lambda x : 1 if x == 'severe' else 0)
        df['hard_pred'] = df['preds'].apply(lambda x : 1 if x >= 0.5 else 0)
        df['patient_name'] = df['patient_id'].apply(lambda x : int(x.split('_')[1]))
        mean_result = df.groupby(['patient_name'])['preds'].mean().to_dict()
        globals()['df_{}'.format(fold_num)]['patient_name'] = globals()['df_{}'.format(fold_num)]['patient_id'].apply(lambda x : int(x.split('_')[1]))
        globals()['df_{}'.format(fold_num)]['patient_level_pred'] = globals()['df_{}'.format(fold_num)]['patient_name'].map(mean_result)
        globals()['df_{}'.format(fold_num)].drop_duplicates(subset=['patient_name', 'patient_level_pred'], inplace=True)

        labels_patient_level = globals()['df_{}'.format(fold_num)]['label'].values
        preds_patient_level = globals()['df_{}'.format(fold_num)]['patient_level_pred'].values

        np.save(fold_path.joinpath('patient_level_preds_ensemble.npy'), preds_patient_level)
        np.save(fold_path.joinpath('patient_level_labels_ensemble.npy'), labels_patient_level)

        soft_preds_patient_level = preds_patient_level.copy()
        hard_preds_patient_level = np.where(soft_preds_patient_level >= 0.5, 1, 0)

        cm_generator = ConfusionMatrix(task = 'binary', num_classes=2)
        confusionmatrix_patient_level = cm_generator(preds = torch.tensor(hard_preds_patient_level), target = torch.tensor(labels_patient_level))

        auc_score_patient_level   = str(round(roc_auc_score(labels_patient_level, soft_preds_patient_level)* 100, 2))
        sensitivity_paitent_level = str(round(recall_score(labels_patient_level, hard_preds_patient_level)    * 100, 2))
        speicifcity_paitnet_level = str(round(int(confusionmatrix_patient_level[0,0])/(int(confusionmatrix_patient_level[0,0])+int(confusionmatrix_patient_level[0,1]))*100, 2))
        accuracy_patient_level    = str(round(accuracy_score(labels_patient_level, hard_preds_patient_level)  * 100, 2))
        nll_patient_level         = str(round(log_loss(labels_patient_level, hard_preds_patient_level, eps=1e-15, normalize=True),2))
        brier_patient_level       = str(round(brier_score_loss(labels_patient_level, hard_preds_patient_level),2))

        fold_logger(f"'ensemble_patient_level'|{auc_score_patient_level.ljust(10, ' ')}| {sensitivity_paitent_level.ljust(10, ' ')}| {speicifcity_paitnet_level.ljust(10, ' ')}| {accuracy_patient_level.ljust(10, ' ')}| {nll_patient_level.ljust(10, ' ')}| {brier_patient_level.ljust(10, ' ')}"  )
        fold_logger(f"\n{confusionmatrix_patient_level}")
        total_scores_patient_level.append([fold_num, auc_score_patient_level, sensitivity_paitent_level, speicifcity_paitnet_level, accuracy_patient_level, nll_patient_level, brier_patient_level])         


    total_scores_image_level = pd.DataFrame(total_scores_image_level, columns = ['fold_num', 'AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'NLL', 'Brier'])
    total_scores_image_level.to_csv(total_path.joinpath('image_level_total_scores.csv'), index = False)
    total_scores_patient_level = pd.DataFrame(total_scores_patient_level, columns = ['fold_num', 'AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'NLL', 'Brier'])
    total_scores_patient_level.to_csv(total_path.joinpath('patient_level_total_scores.csv'), index = False)

