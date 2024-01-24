# """
# In the step 3, we will try to use t-SNE and PCA methods to visualize chemspace.
# If you want to use the more descriptors, other cheminformatics package can be used, such as CDK and Openbabel.
# Herein, the rdkit descriptors will be used.
# """
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

################################################
warnings.filterwarnings('ignore')
# global variable
file_path = './output_file/features_rdkit.csv'

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

################################################
# input data
data = pd.read_csv(file_path, index_col='Unnamed: 0')
names = data.columns

# normalized the feature data
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=names)

################################################
# using PCA to visualize the chemspace
pca = PCA()  # only save the first two components
transform_data = pca.fit_transform(data)

explain = pca.explained_variance_ratio_  # output the explained variance ratios of PCs

def top20pcs(array):
    df_explain = pd.DataFrame(array)
    df_explain.columns = ['explained variance ratios']
    print(df_explain.head())
    cutoff = int(len(df_explain) / 5)
    df_top20 = df_explain.head(cutoff)

    return df_top20


df_pca20 = top20pcs(explain)  # visualize the top 20% pc
plt.figure(figsize=(10, 8))
plt.plot(df_pca20.index, df_pca20['explained variance ratios'])
plt.title("Using PCA to visualize the chemspace")
plt.xlabel("PCs")
plt.ylabel("Explained variance ratios")
plt.grid(True)
plt.show()


loadings = pca.components_  # output the loading of PCs, suggest only pc1 and pc2

pc1_loadings_squared_sum = np.sum(loadings[0]**2)
pc2_loadings_squared_sum = np.sum(loadings[1]**2)
print("The squared sum of PC1：", pc1_loadings_squared_sum)
print("The squared sum of PC2：", pc2_loadings_squared_sum)
pc1_loading = loadings[0]**2
pc1_loading = pc1_loading.T
pc2_loading = loadings[1]**2
pc2_loading = pc2_loading.T

# the results of loading analysis
loading_matrix1 = pd.DataFrame(pc1_loading, index=data.columns, columns=['PC1'])
loading_matrix2 = pd.DataFrame(pc2_loading, index=data.columns, columns=['PC2'])
print(loading_matrix1)
print(loading_matrix2)

loading_matrix1.to_csv('./output_file/loading-f-pc1.csv')
loading_matrix2.to_csv('./output_file/loading-f-pc2.csv')

df_pca = pd.DataFrame()
df_pca['PCA-PC1'] = transform_data[:, 0]
df_pca['PCA-PC2'] = transform_data[:, 1]
df_pca['PCA-PC3'] = transform_data[:, 2]
df_pca.to_csv('./output_file/PCA_results.csv', index=False)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA-PC1', y='PCA-PC2', data=df_pca)
plt.title('Chemspace visualization by PCA')
plt.xlabel('Principal Component 1', fontdict={'fontsize': 12})
plt.ylabel('Principal Component 2', fontdict={'fontsize': 12})
plt.show()

################################################
tsne = TSNE(n_components=3, random_state=0)
transform_data_tsne = tsne.fit_transform(data)

df_tsne = pd.DataFrame()
df_tsne['TSNE-PC1'] = transform_data_tsne[:, 0]
df_tsne['TSNE-PC2'] = transform_data_tsne[:, 1]
df_tsne['TSNE-PC3'] = transform_data_tsne[:, 2]
df_tsne.to_csv('./output_file/tSNE_results.csv', index=False)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE-PC1', y='TSNE-PC2', data=df_tsne)

plt.title('Chemspace visualization by t-SNE')
plt.xlabel('t-SNE Component 1', fontdict={'fontsize': 12})
plt.ylabel('t-SNE Component 2', fontdict={'fontsize': 12})
plt.show()

################################################
end_time = datetime.datetime.now()
print('End running，time：：', end_time.strftime('%Y-%m-%d %H:%M:%S'))