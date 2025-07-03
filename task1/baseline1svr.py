# 首先，导入库
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger,Chem
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
RDLogger.DisableLog('rdApp.*')

def mfgen(mol, nBits=2048, radius=2):
    '''
    Parameters
    ----------
    mol : mol
        RDKit mol object.
    nBits : int
        Number of bits for the fingerprint.
    radius : int
        Radius of the Morgan fingerprint.
    Returns
    -------
    mf_desc_map : ndarray
        ndarray of molecular fingerprint descriptors.
    '''
    # 返回分子的位向量形式的Morgan fingerprint
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(list(map(eval, list(fp.ToBitString()))))


# 加载数据
def vec_cpd_lst(smi_lst):
    smi_set = list(set(smi_lst))
    smi_vec_map = {}
    for smi in tqdm(smi_set):  # tqdm：显示进度条
        mol = Chem.MolFromSmiles(smi)
        smi_vec_map[smi] = mfgen(mol)
    smi_vec_map[''] = np.zeros(2048)

    vec_lst = [smi_vec_map[smi] for smi in smi_lst]
    return np.array(vec_lst)

# 定义SVR的参数网格
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
    'epsilon': [0.01, 0.1, 0.2, 0.5]
}

dataset_dir = 'dataset'

train_df = pd.read_csv(f'{dataset_dir}/round1_train_data.csv')
test_df = pd.read_csv(f'{dataset_dir}/round1_test_data.csv')

print(f'Training set size: {len(train_df)}, test set size: {len(test_df)}')

# 从csv中读取数据
train_rct1_smi = train_df['Reactant1'].to_list()
train_rct2_smi = train_df['Reactant2'].to_list()
train_add_smi = train_df['Additive'].to_list()
train_sol_smi = train_df['Solvent'].to_list()

# 将SMILES转化为分子指纹
train_rct1_fp = vec_cpd_lst(train_rct1_smi)
train_rct2_fp = vec_cpd_lst(train_rct2_smi)
train_add_fp = vec_cpd_lst(train_add_smi)
train_sol_fp = vec_cpd_lst(train_sol_smi)
# 在dim=1维度进行拼接。即：将一条数据的Reactant1,Reactant2,Product,Additive,Solvent字段的morgan fingerprint拼接为一个向量。
train_x = np.concatenate([train_rct1_fp,train_rct2_fp,train_add_fp,train_sol_fp],axis=1)
train_y = train_df['Yield'].to_numpy()

# 测试集也进行同样的操作
test_rct1_smi = test_df['Reactant1'].to_list()
test_rct2_smi = test_df['Reactant2'].to_list()
test_add_smi = test_df['Additive'].to_list()
test_sol_smi = test_df['Solvent'].to_list()

test_rct1_fp = vec_cpd_lst(test_rct1_smi)
test_rct2_fp = vec_cpd_lst(test_rct2_smi)
test_add_fp = vec_cpd_lst(test_add_smi)
test_sol_fp = vec_cpd_lst(test_sol_smi)
test_x = np.concatenate([test_rct1_fp,test_rct2_fp,test_add_fp,test_sol_fp],axis=1)

# # Model fitting
# model = RandomForestRegressor(n_estimators=10,max_depth=10,min_samples_split=2,min_samples_leaf=1,n_jobs=-1) # 实例化模型，并指定重要参数
# model.fit(train_x,train_y) # 训练模型

# 实例化SVR模型
svr = SVR()

# 实例化GridSearchCV
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, n_jobs=-1, verbose=2)

# 执行网格搜索
grid_search_svr.fit(train_x, train_y)

# 输出最佳参数组合
print(f'Best parameters found: {grid_search_svr.best_params_}')
# 使用最佳参数组合训练模型
best_rf = grid_search_svr.best_estimator_
best_rf.fit(train_x, train_y)

# 保存模型
with open('./random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

# 加载模型
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 预测\推理
test_pred = loaded_model.predict(test_x)

ans_str_lst = ['rxnid,Yield']
for idx,y in enumerate(test_pred):
    ans_str_lst.append(f'test{idx+1},{y:.4f}')
with open('./submit.txt','w') as fw:
    fw.writelines('\n'.join(ans_str_lst))