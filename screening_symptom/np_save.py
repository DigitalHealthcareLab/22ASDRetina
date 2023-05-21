from pathlib import Path
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

total_path = Path('/home/jaehan0605/MAIN_ASDfundus/ResNeXt50_TD_seed_2023_Total')

for fold_num in range(10):
    fold_path = total_path.joinpath(f'fold_{fold_num}')
    inference_result_df = pd.read_csv(fold_path.joinpath('inference_result.csv'))
    for model_num in range(5):
        df_for_each_model = inference_result_df[['severity','image_stem', f'pred_{model_num}']]
        df_for_each_model['patient_name'] = df_for_each_model['image_stem'].apply(lambda x: int(x.split('_')[1]))
        mean_result = df_for_each_model.groupby(['patient_name'])[f'pred_{model_num}'].mean().to_dict()
        df_for_each_model['patient_name'] = df_for_each_model['image_stem'].apply(lambda x: int(x.split('_')[1]))
        df_for_each_model['patient_level_pred'] = df_for_each_model['patient_name'].map(mean_result)
        df_for_each_model.drop_duplicates(subset = ['patient_name', 'patient_level_pred'], inplace = True)
        patient_level_preds = np.array(df_for_each_model['patient_level_pred'])

        patient_level_labels = np.array(df_for_each_model['severity'])

        np.save(fold_path.joinpath(f'patient_level_preds_singlemodel_{model_num}.npy'), patient_level_preds)
        np.save(fold_path.joinpath(f'patient_level_labels_singlemodel.npy'), patient_level_labels)

    df_ensemble = inference_result_df[['severity','image_stem', 'deep_ensemble']]
    df_ensemble['patient_name'] = df_ensemble['image_stem'].apply(lambda x: int(x.split('_')[1]))
    mean_result = df_ensemble.groupby(['patient_name'])['deep_ensemble'].mean().to_dict()
    df_ensemble['patient_name'] = df_ensemble['image_stem'].apply(lambda x: int(x.split('_')[1]))
    df_ensemble['patient_level_pred'] = df_ensemble['patient_name'].map(mean_result)
    df_ensemble.drop_duplicates(subset = ['patient_name', 'patient_level_pred'], inplace = True)

    ensemble_preds = np.array(df_ensemble['patient_level_pred'], )
    ensemble_labels = np.array(df_ensemble['severity'])

    np.save(fold_path.joinpath(f'patient_level_preds_ensemble.npy'), ensemble_preds)
    np.save(fold_path.joinpath(f'patient_level_labels_ensemble.npy'), ensemble_labels)
