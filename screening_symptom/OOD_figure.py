import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

OOD = np.load('/home/jaehan0605/MAIN_ASDfundus/OOD_ADOS.npy')
TEST = np.load('/home/jaehan0605/MAIN_ASDfundus/test_ADOS.npy')

plt.clf()

plt.figure(figsize=(5,4), dpi=200)
sns.kdeplot(data = [OOD, TEST], common_norm=True, common_grid=True, gridsize=1000)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.legend(["Test set", "OOD set"])
plt.savefig("/home/jaehan0605/MAIN_ASDfundus/Figure.svg", dpi=300, format = 'svg')
plt.savefig("/home/jaehan0605/MAIN_ASDfundus/Figure.png", dpi=300, format = 'png')


OOD.mean()
OOD.std()

TEST.mean()
TEST.std()

statistic, p_value = mannwhitneyu(OOD, TEST)
p_value

