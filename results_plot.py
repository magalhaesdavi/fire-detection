import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns

authors = ['Rudz', 'Chen', 'Celik', 'BoWFire', 'Color classification',
           'CNNFire T = 0.40', 'CNNFire T = 0.45', 'CNNFire T = 0.50', 'Mlích', 'Our method']
tpr = [0.41, 0.106, 0.63, 0.65, 0.77, 0.821, 0.855, 0.89, 0.8742, 0.91]
fpr = [0.05, 0.1063, 0.10, 0.03, 0.13, 0.025, 0.048, 0.075, 0.016, 0.06]

df = pd.DataFrame(index=[0])
df_list = []
for i in range(len(authors)):
    d = {'Author': authors[i], 'TPR': tpr[i], 'FPR': fpr[i]}
    df = pd.DataFrame(d, index=[0])
    df_list.append(df)

df = pd.concat(df_list)
# display(df)

# plt.figure(figsize=(12,8))
plt.figure(figsize=(16,10))
sns.set(font_scale=1.2, style="whitegrid")
roc_plot = sns.scatterplot(data=df, x="FPR", y="TPR", hue='Author', style="Author", s=500)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='20',
           title_fontsize='20', markerscale=2.6, title='Author')
fig = roc_plot.get_figure()
fig.savefig("images/result_comparison.png", bbox_inches="tight")