# 导入库
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger, Chem
import numpy as np
import cuml
from cuml.svm import SVR as cuSVR
from cuml.model_selection import GridSearchCV as cuGridSearchCV
from cuml.dask.common.utils import persist_across_workers
from cuml.dask.common import to_output_type
from cuml.dask.svm import SVR as cuSVR_dask
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
RDLogger.DisableLog('rdApp.*')

def mfgen(mol, nBits=2048, radius=2):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(list(map(eval, list(fp.ToBitString()))))

# 加载数据
def vec_cpd_lst(smi_lst):
    smi_set = list(set(smi_lst))
    smi_vec_map = {}
    for smi in tqdm(smi_set):
        mol = Chem.MolFromSmiles(smi)
        smi_vec_map[smi] = mfgen(mol)
    smi_vec_map[''] = np.zeros(2048)

    vec_lst = [smi_vec_map[smi] for smi in smi_lst]
    return np.array(vec_lst)

# 定义SVR的参数网格
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'] + list(np.logspace(-4, 4, 9)),
    'epsilon': [0.001, 0.01, 0.1, 1]
}

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
train_x = np.concatenate([train_rct1_fp, train_rct2_fp, train_add_fp, train_sol_fp], axis=1).astype(np.float32)
train_y = train_df['Yield'].to_numpy().astype(np.float32)

test_rct1_smi = test_df['Reactant1'].to_list()
test_rct2_smi = test_df['Reactant2'].to_list()
test_add_smi = test_df['Additive'].to_list()
test_sol_smi = test_df['Solvent'].to_list()

test_rct1_fp = vec_cpd_lst(test_rct1_smi)
test_rct2_fp = vec_cpd_lst(test_rct2_smi)
test_add_fp = vec_cpd_lst(test_add_smi)
test_sol_fp = vec_cpd_lst(test_sol_smi)
test_x = np.concatenate([test_rct1_fp, test_rct2_fp, test_add_fp, test_sol_fp], axis=1).astype(np.float32)

# 初始化Dask CUDA集群
cluster = LocalCUDACluster()
client = Client(cluster)

# 使用Dask的分布式SVR和GridSearchCV
svr = cuSVR_dask()
grid_search_svr = cuGridSearchCV(svr, param_grid_svr, cv=10, verbose=2)

# 将数据转换为Dask数组
train_x_dask = client.scatter(train_x)
train_y_dask = client.scatter(train_y)

# 执行分布式网格搜索
grid_search_svr.fit(train_x_dask, train_y_dask)

# 输出最佳参数组合
print(f'Best parameters found: {grid_search_svr.best_params_}')
# 使用最佳参数组合训练模型
best_svr = grid_search_svr.best_estimator_
best_svr.fit(train_x_dask, train_y_dask)

# 保存模型
with open('./svr_model.pkl', 'wb') as file:
    pickle.dump(best_svr, file)

# 预测
test_x_dask = client.scatter(test_x)
test_pred = best_svr.predict(test_x_dask)
test_pred = to_output_type(test_pred)

ans_str_lst = ['rxnid,Yield']
for idx, y in enumerate(test_pred):
    ans_str_lst.append(f'test{idx+1},{y:.4f}')
with open('./submit_svr.txt', 'w') as fw:
    fw.writelines('\n'.join(ans_str_lst))

# 关闭Dask客户端
client.close()
