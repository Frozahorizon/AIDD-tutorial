# """
# In the step 2.5, we will try to generate several molecular fingerprints from rdkit package.
# If you want to use the more fingerprint, other cheminformatics package can be used, such as CDK and Openbabel.
# Herein, Morgan fingerprint / MACCS fingerprints / AtomPair fingerprints will be used.
# """
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

################################################
warnings.filterwarnings('ignore')
# global variable
file_path = './output_file/clean.csv'
radius = 3
nbits = 1024
bit_info = []

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

################################################
# import data

data = pd.read_csv(file_path)

mols = []
for smile in tqdm(data['Smiles'], desc="Converting", unit="mol"):
    mols.append(Chem.MolFromSmiles(smile))


################################################
# """
# 1. define a func to create morgan fingerprint
# 2. apply this func on all molecules
# 3. output the fingerprint file as csv
# """

def morgan_fingerprint(molecule, radius, nbits):
    bit = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nbits, bitInfo=bit)
    fp_array = np.zeros((1,), dtype=np.int8)
    ConvertToNumpyArray(fp, fp_array)
    df = pd.DataFrame(fp_array)

    return df, bit


df_morgan = pd.DataFrame()
for mol in tqdm(mols, desc="Calculating", unit="mol"):
    df_fp, bi = morgan_fingerprint(mol, radius=radius, nbits=nbits)
    bit_info.append(bi)
    df_morgan = pd.concat([df_morgan, df_fp], axis=1)

df_morgan = df_morgan.T
print('The shape of morgan fingerprint', df_morgan.shape)
df_morgan.index = range(len(df_morgan))
df_morgan.to_csv('./output_file/morgan.csv', index=False)


################################################
# """
# 1. define a func to create MACCS fingerprint
# 2. apply this func on all molecules
# 3. output the fingerprint file as csv
# """
def maccs_fingerprint(molecule):
    mfp = MACCSkeys.GenMACCSKeys(molecule)
    mfp_arr = np.array(mfp, dtype=object)
    df = pd.DataFrame(mfp_arr).T
    mkey = mfp.GetOnBits()
    mkey_arr = np.array(mkey, dtype=object)

    return df, mkey_arr


maccs_dict = []
df_maccs = pd.DataFrame()
for mol in tqdm(mols, desc="Calculating", unit="mol"):
    df_ms, maccs_key = maccs_fingerprint(mol)
    maccs_dict.append(maccs_key)
    df_maccs = pd.concat([df_maccs, df_ms], axis=0)

df_maccs.index = range(len(df_maccs))
print('The shape of maccs fingerprint', df_maccs.shape)
df_maccs.to_csv('./output_file/maccs.csv', index=False)


################################################
# """
# 1. define a func to create AtomPair fingerprint
# 2. apply this func on all molecules
# 3. output the fingerprint file as csv
# """
def atompair_fingerprint(molecule, nBits):
    afp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(molecule, nBits=nBits)
    arr = np.zeros((1,))
    ConvertToNumpyArray(afp, arr)
    df = pd.DataFrame(arr)
    return df


df_atom = pd.DataFrame()
for mol in tqdm(mols, desc="Calculating", unit="mol"):
    df_afp = atompair_fingerprint(mol, nBits=nbits)
    df_atom = pd.concat([df_atom, df_afp], axis=1)

df_atom = df_atom.T
df_atom.index = range(len(df_atom))
print('The shape of AtomPair fingerprint', df_atom.shape)
df_atom.to_csv('./output_file/atompair.csv', index=False)

################################################
end_time = datetime.datetime.now()
print('Start running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))