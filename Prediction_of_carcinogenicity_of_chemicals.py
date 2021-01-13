#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing the necessary packages

# In[2]:


get_ipython().system(' pip install cirpy')
import cirpy
import sys 
sys.path.append('/usr/local/lib/python3.7/site-packages/')
get_ipython().system(' wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local')
get_ipython().system(' conda install -c rdkit rdkit -y')

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# **Reading the dataset**

# In[3]:


df = pd.read_excel('/kaggle/input/803-cpdb/final_cd.xlsx')
import gc
df.drop(columns=['ID_v5','  CASRN'],inplace=True)
df['SMILES'].replace(['Cbr'],'CBr',inplace=True)
df=df[df['SMILES'].notna()]
gc.collect()
df.shape


# In[4]:


df['Chemical Name']=='Acetaldehyde'


# In[5]:


df


# In[6]:


potency={'NP':0,'P':1}
df['Carcinogenic Potency Expressed as P or NP']=df['Carcinogenic Potency Expressed as P or NP'].map(potency).astype('int')


# **Calculation of descriptors**

# In[7]:


mol_lst=[]

for i in df.SMILES:
    mol=Chem.MolFromSmiles(i)
    mol_lst.append(mol) # Calculation of Mol Objects

desc_lst=[i[0] for i in Descriptors._descList]
descriptor=MoleculeDescriptors.MolecularDescriptorCalculator(desc_lst)
descrs = [] #Calculation of descriptors

for i in range(len(mol_lst)):
    descrs.append(descriptor.CalcDescriptors(mol_lst[i]))
molDes=pd.DataFrame(descrs,columns=desc_lst)
molDes.head(5)


# In[8]:


rfc = RandomForestClassifier(max_depth=9, min_samples_leaf=3, min_samples_split=5,
                       n_estimators=50)


# In[9]:


rf_feat = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt',
            'ExactMolWt', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
           'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI',
            'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
            'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
            'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1',
            'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
           'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA3', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA2',
            'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA2',
            'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
           'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6','VSA_EState7',
           'VSA_EState8', 'FractionCSP3', 'NumHDonors', 'NumRotatableBonds', 'MolLogP', 'MolMR', 'fr_NH0', 'fr_nitroso']


# In[10]:


rfc.fit(molDes[rf_feat],df['Carcinogenic Potency Expressed as P or NP'])


# In[ ]:


# import sklearn.external.joblib as extjoblib
import joblib


# In[ ]:


joblib.dump(rfc, 'model_rfc')


# In[14]:


number = int(input('Enter: '))
chem_lst = []
for i in range(0,number):
    print('Chemical number',i+1 )
    chemical  = input('Enter : ')
    chem_lst.append(chemical)
smiles = []
for i in chem_lst:
    che = cirpy.resolve(i,'smiles')
    smiles.append(che)
# print('The smiles format of {} is {}'.format(chemical,che))
mol = []
for i in smiles:
    mol_che = Chem.MolFromSmiles(i)
    mol.append(mol_che)
l = []
for i in mol:
    des_che = descriptor.CalcDescriptors(i)
    l.append(des_che)
features = pd.DataFrame(l,columns = desc_lst)
arr = rfc.predict(features[rf_feat])
prob = rfc.predict_proba(features[rf_feat])

r = []
for i in arr:
    if i == 0:
        r.append('Non-Carcinogen')
    else:
        r.append('Carcinogen')
results = pd.DataFrame()
results['Chemical'] = chem_lst
results['Likely to be'] = r
results['Percentage'] = prob[:,1]
results


# In[ ]:




