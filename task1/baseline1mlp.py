import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam

def mfgen(mol, nBits=2048, radius=2):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(list(map(int, list(fp.ToBitString()))))

def vec_cpd_lst(smi_lst):
    smi_set = list(set(smi_lst))
    smi_vec_map = {}
    for smi in tqdm(smi_set):
        mol = Chem.MolFromSmiles(smi)
        smi_vec_map[smi] = mfgen(mol)
    smi_vec_map[''] = np.zeros(2048)
    vec_lst = [smi_vec_map[smi] for smi in smi_lst]
    return np.array(vec_lst)

# 加载数据
dataset_dir = 'dataset'
train_df = pd.read_csv(f'{dataset_dir}/round1_train_data.csv')
test_df = pd.read_csv(f'{dataset_dir}/round1_test_data.csv')

print(f'Training set size: {len(train_df)}, test set size: {len(test_df)}')

train_rct1_smi = train_df['Reactant1'].to_list()
train_rct2_smi = train_df['Reactant2'].to_list()
train_add_smi = train_df['Additive'].to_list()
train_sol_smi = train_df['Solvent'].to_list()

train_rct1_fp = vec_cpd_lst(train_rct1_smi)
train_rct2_fp = vec_cpd_lst(train_rct2_smi)
train_add_fp = vec_cpd_lst(train_add_smi)
train_sol_fp = vec_cpd_lst(train_sol_smi)
train_x = np.concatenate([train_rct1_fp, train_rct2_fp, train_add_fp, train_sol_fp], axis=1)
train_y = train_df['Yield'].to_numpy()

test_rct1_smi = test_df['Reactant1'].to_list()
test_rct2_smi = test_df['Reactant2'].to_list()
test_add_smi = test_df['Additive'].to_list()
test_sol_smi = test_df['Solvent'].to_list()

test_rct1_fp = vec_cpd_lst(test_rct1_smi)
test_rct2_fp = vec_cpd_lst(test_rct2_smi)
test_add_fp = vec_cpd_lst(test_add_smi)
test_sol_fp = vec_cpd_lst(test_sol_smi)
test_x = np.concatenate([test_rct1_fp, test_rct2_fp, test_add_fp, test_sol_fp], axis=1)

# 标准化
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# 定义神经网络模型
def create_model(neurons=64, dropout_rate=0.5, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

param_grid = {
    'neurons': [64, 128, 256],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150],
    'learning_rate': [0.01, 0.001, 0.0001],
    'dropout_rate': [0.2, 0.3, 0.5]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_x, train_y)

print(f'Best parameters found: {grid_result.best_params_}')
best_model = grid_result.best_estimator_

best_model.fit(train_x, train_y)

# 保存模型
best_model.model.save('best_mlp_model.h5')

# 加载模型
from tensorflow.keras.models import load_model
loaded_model = load_model('best_mlp_model.h5')

# 预测
test_pred = loaded_model.predict(test_x)

ans_str_lst = ['rxnid,Yield']
for idx, y in enumerate(test_pred):
    ans_str_lst.append(f'test{idx+1},{y[0]:.4f}')
with open('./submit_mlp.txt', 'w') as fw:
    fw.writelines('\n'.join(ans_str_lst))
