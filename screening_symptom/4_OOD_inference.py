from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,log_loss
from torch.nn.functional import softmax
import torch
import warnings
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import math


from src.dataloader_setter_OOD import Dataloader_Setter
from src.logger import print_logger
from src.load_model import load_model
from src.data_splitting import make_dataframe_OOD
from src.seed import seed_everything

seed_everything(42)

# 오류 경고 무시하기
warnings.filterwarnings(action='ignore')
# 오류 메세지 다시 보이게 하기
# warnings.filterwarnings(action='default')


#### 변경이 필요한 부분
batch_size = 1
device = 'cuda'
initial_learning_rate = 0.001

save_path = Path('/home/jaehan0605/MAIN_ASDfundus/OOD_images/Processed_dataset/')

make_dataframe_OOD(save_path)

####
log_name = Path('/home/jaehan0605/MAIN_ASDfundus/ResNeXt50_TD_seed_2023_ADOS')
total_path = Path(log_name)

logger = print_logger(total_path, "Total_Inference.txt")
dataloader = Dataloader_Setter(logger)

df = pd.read_csv('/home/jaehan0605/MAIN_ASDfundus/OOD_images/Processed_dataset/dataset.csv').query('FOLD_0==2')[['image_path']].reset_index(drop=True)

if __name__ == "__main__" : 

    fold_dataloader = dataloader.load(0, batch_size)
    test_dataloader = fold_dataloader.get("TEST")
    logger(f'TEST Dataloader : {test_dataloader.dataset.__len__()}', end = '\n')

    full_entropy_list_OOD = []
    full_entropy_list_TESTSET = []

    for fold_num in range(10) :         
        fold_path = total_path.joinpath(f'fold_{fold_num}')
        fold_logger = print_logger(fold_path, "OOD_inference_log.txt")
        
        preds_ensemble = [] #List for deep ensemble in retinal fundus image level
        preds_ensemble_patient = [] #List for deep ensemble in patient level
        globals()['entropy_list_OOD_{}'.format(fold_num)] = [] #List for OOD entropy in patient level
        

        # Load Trained Model
        model = load_model(model_name='ResNeXt50', pretrained=True, num_classes=2)

        globals()['df_{}'.format(fold_num)] = pd.DataFrame()

        print(f'OOD inference for fold_{fold_num} started')
        
        for model_num in range(5):
            state_dict = torch.load(fold_path.joinpath(f'model_{model_num}.pth'))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # inference
            with torch.no_grad():
                preds = []
                patient_id = []

                for X, path_stem in test_dataloader:
                    X = X.to(device)
                    pred = model(X)
                    preds.extend(softmax(pred)[:,1].cpu().numpy())
                    patient_id.extend(path_stem)
                
                list_preds = preds
                preds = np.array(preds)
                preds_ensemble.append(list_preds)

                globals()['df_{}'.format(fold_num)]['patient_id'] = patient_id
                globals()['df_{}'.format(fold_num)][f'model_{model_num}'] = list_preds
               
        #######ensemble results at image level (OOD)
        preds_df_folds = pd.DataFrame(preds_ensemble)
        preds_ensemble_list = []

        for col in preds_df_folds.columns:
            p_for_severe = preds_df_folds[col].mean()
            preds_ensemble_list.append(p_for_severe)

        preds_en = np.array(preds_ensemble_list)

        for i in range(len(preds_en)):
            p = preds_en[i]
            entropy = -p*math.log2(p) - (1-p)*math.log2(1-p+ 0.00000000001)
            full_entropy_list_OOD.append(entropy)
            globals()['entropy_list_OOD_{}'.format(fold_num)].append(entropy)
        
        #############Entropy of test set
        
        globals()['entropy_list_TESTSET_{}'.format(fold_num)] = []

        fold_preds = list(np.load(fold_path.joinpath(f"patient_level_preds_ensemble.npy")))
        for j in range(len(fold_preds)):
            p = fold_preds[j]
            entropy = -p*math.log2(p) - (1-p)*math.log2(1-p+ 0.00000000001)
            full_entropy_list_TESTSET.append(entropy)
            globals()['entropy_list_TESTSET_{}'.format(fold_num)].append(entropy)

        ###### Plotting
        # OOD set : entropy_list_OOD_{fold_num} / full_entropy_list_OOD
        # TEST set : entropy_list_TESTSET_{fold_num} / full_entropy_list_TESTSET

        sns.kdeplot(globals()['entropy_list_OOD_{}'.format(fold_num)]) ##ood ploting
        sns.kdeplot(globals()['entropy_list_TESTSET_{}'.format(fold_num)]) ##test set plotting
        data1 = np.array(globals()['entropy_list_OOD_{}'.format(fold_num)])
        data2 = np.array(globals()['entropy_list_TESTSET_{}'.format(fold_num)])
        
        # sns.histplot(data=data1, stat='probability', kde=True, binwidth=0.05, alpha=0.5, edgecolor = 'none')
        # sns.histplot(data=data2, stat='probability', kde=True, binwidth=0.05, alpha=0.5, edgecolor = 'none')

        # sns.displot(data = [data1, data2], kind = 'kde', height=5)

        # plt.xlim([-0.1,1])
        # plt.ylim([0,50])
        plt.xlabel("Entropy")
        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylabel("Density")
        plt.legend(["OOD set", "Test set"])
        plt.savefig(f"/home/jaehan0605/MAIN_ASDfundus/OOD_images/main_figures/Entropy_figure_10fold_{fold_num}.png")
        plt.clf()
    
    # sns.kdeplot(full_entropy_list_OOD) ##ood ploting
    # sns.kdeplot(full_entropy_list_TESTSET) ##test set plotting

    data1 = np.array(full_entropy_list_OOD)
    data2 = np.array(full_entropy_list_TESTSET)


    np.save(Path('/home/jaehan0605/MAIN_ASDfundus/OOD.npy'), data1)
    np.save(Path('/home/jaehan0605/MAIN_ASDfundus/test.npy'), data2)

    # sns.histplot(data=data1, stat='probability', kde=True, binwidth=0.05, alpha=0.5, edgecolor = 'none')
    # sns.histplot(data=data2, stat='probability', kde=True, binwidth=0.05, alpha=0.5, edgecolor = 'none')

    sns.displot(data = [data1, data2], kind = 'kde', height=5)

    plt.xlim([-0.1,0.5])
    plt.ylim(0,50)
    plt.xlabel("Entropy")
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel("Density")
    plt.legend(["OOD set", "Test set"])
    plt.savefig(f"/home/jaehan0605/MAIN_ASDfundus/OOD_images/main_figures/Entropy_figure_Fullset.png")
    plt.clf()





