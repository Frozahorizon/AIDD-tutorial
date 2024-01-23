import pandas as pd
import warnings
import datetime
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

################################################
warnings.filterwarnings('ignore')
# global variable
file_path = './output_file/clean.csv'

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

################################################
# input data
data = pd.read_csv(file_path)
# data = pd.read_excel(file_path)

# """
# Converting the smiles to mol
# """
mols = []
for mol in tqdm(data['Smiles'], desc="Converting", unit="mol"):
    mols.append(Chem.MolFromSmiles(mol))


# """
# herein, we used feature of rdkit, if you want to use more descriptors,
# you can search the Chem web in google
# """
def calcMoldes(mol):
    des_dict = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_dict)
    tuple_mol = calculator.CalcDescriptors(mol)
    df_des = pd.DataFrame(tuple_mol).T
    df_des.columns = des_dict

    return df_des

# """
# generate the ML features
# """
fea_dict = [x[0] for x in Descriptors._descList]
df_feature = pd.DataFrame(columns=fea_dict)
for mol in tqdm(mols, desc="Calculating",
                unit="mol"):
    df_des = calcMoldes(mol)
    df_feature = pd.concat([df_feature, df_des], axis=0)

df_feature.index = range(len(df_feature))

df_feature.to_csv('./output_file/features_rdkit.csv')

end_time = datetime.datetime.now()
print('End running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))