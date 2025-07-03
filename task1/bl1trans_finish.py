import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertConfig
from bayes_opt import BayesianOptimization
from rdkit import RDLogger,Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取分子的位向量形式的Morgan fingerprint
def mfgen(mol, nBits=2048, radius=2):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(list(map(int, list(fp.ToBitString()))))

# 处理SMILES字符串列表，生成分子指纹
def vec_cpd_lst(smi_lst):
    smi_set = list(set(smi_lst))
    smi_vec_map = {}
    for smi in tqdm(smi_set):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi_vec_map[smi] = mfgen(mol)
        else:
            smi_vec_map[smi] = np.zeros(2048)
    return np.array([smi_vec_map[smi] for smi in smi_lst])

# 加载数据
# train_df = pd.read_csv('dataset/round1_train_data.csv')
train_df = pd.read_csv('dataset/round1_train_data.csv')
test_df = pd.read_csv('dataset/round1_test_data.csv')

# # 检查是否存在 NaN 值
# nan_check = train_df.isnull().sum().sum()
# # 检查是否存在无穷大值
# inf_check = train_df.isin([np.Inf, -np.Inf]).sum().sum()

# if nan_check > 0 or inf_check > 0:
#     # 填充缺失值为列的均值
#     train_df = train_df.fillna(data.mean())
#     # 检查和处理无穷大值
#     train_df = train_df.replace([np.inf, -np.inf], np.nan)
#     # 删除仍然包含 NaN 值的行
#     train_df = train_df.dropna()

# 检查数据类型和数据范围
# 例如，将产率列转换为浮点型并检查是否有无效数据
train_df['Yield'] = train_df['Yield'].astype(float)

# # 检查数据是否超出范围
# data_check = np.isfinite(train_df).all().all()
# if not data_check:
#     # 如果仍有问题的数据，进行处理（如删除或替换）
#     train_df = train_df[np.isfinite(train_df).all(axis=1)]

# 处理训练数据
train_rct1_fp = vec_cpd_lst(train_df['Reactant1'].to_list())
train_rct2_fp = vec_cpd_lst(train_df['Reactant2'].to_list())
train_add_fp = vec_cpd_lst(train_df['Additive'].to_list())
train_sol_fp = vec_cpd_lst(train_df['Solvent'].to_list())
train_x = np.concatenate([train_rct1_fp, train_rct2_fp, train_add_fp, train_sol_fp], axis=1)
train_y = train_df['Yield'].to_numpy()
# train_y = train_y.reset_index(drop=True)

# 处理测试数据
test_rct1_fp = vec_cpd_lst(test_df['Reactant1'].to_list())
test_rct2_fp = vec_cpd_lst(test_df['Reactant2'].to_list())
test_add_fp = vec_cpd_lst(test_df['Additive'].to_list())
test_sol_fp = vec_cpd_lst(test_df['Solvent'].to_list())
test_x = np.concatenate([test_rct1_fp, test_rct2_fp, test_add_fp, test_sol_fp], axis=1)

# # 检查并移除无效的y值
# def clean_y(x, y):
#     valid_mask = np.isfinite(y) & (y < np.finfo(np.float64).max)
#     return x[valid_mask], y[valid_mask]

# train_x, train_y = clean_y(train_x, train_y)

# 划分训练集和验证集
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

# # 检查验证集的无效值
# val_x, val_y = clean_y(val_x, val_y)

# 转换为PyTorch张量
train_x, train_y = torch.tensor(train_x).float(), torch.tensor(train_y).float()
val_x, val_y = torch.tensor(val_x).float(), torch.tensor(val_y).float()

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=16)

class TransformerRegressor(nn.Module):
    def __init__(self, input_size=8192, seq_length=128, hidden_size=768, num_layers=12, num_attention_heads=12):
        super(TransformerRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, seq_length * hidden_size)
        self.bert = BertModel(BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads))
        self.fc2 = nn.Linear(hidden_size, 1)
        self.seq_length = seq_length
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.seq_length, self.hidden_size)
        outputs = self.bert(inputs_embeds=x)
        pooled_output = outputs.pooler_output
        output = self.fc2(pooled_output)
        return output

# 找到的最佳参数
best_params = {
    'hidden_size': 872.2385000847572,
    'lr': 2.0837516691838533e-05,
    'num_attention_heads': 5.837529409048855,
    'num_layers': 11.724710833451594
}

print(f'Best parameters found: {best_params}')

# 使用最佳参数训练完整模型
hidden_size = int(best_params['hidden_size'])
num_attention_heads = int(best_params['num_attention_heads'])
# 确保 hidden_size 是 num_attention_heads 的整数倍
if hidden_size % num_attention_heads != 0:
    hidden_size = (hidden_size // num_attention_heads) * num_attention_heads
    # 确保 hidden_size 不为 0
    if hidden_size == 0:
        hidden_size = num_attention_heads

best_model = TransformerRegressor(
    hidden_size=hidden_size,
    num_layers=int(best_params['num_layers']),
    num_attention_heads=num_attention_heads
).to(device)

optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['lr'])
loss_fn = nn.MSELoss()

best_model.train()
for epoch in range(5):
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = best_model(x_batch)
        loss = loss_fn(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

# 保存最佳模型
model_path = './best_transformer_model_test.pth'
torch.save(best_model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# 使用最佳模型进行预测
best_model.eval()
test_x_tensor = torch.tensor(test_x).float().to(device)
test_predictions = []
with torch.no_grad():
    for i in range(len(test_x_tensor)):
        output = best_model(test_x_tensor[i:i+1])
        test_predictions.append(output.item())

# 保存预测结果
ans_str_lst = ['rxnid,Yield']
for idx, y in enumerate(test_predictions):
    ans_str_lst.append(f'test{idx + 1},{y:.4f}')
with open('./submit_trans_test.txt', 'w') as fw:
    fw.writelines('\n'.join(ans_str_lst))
print(f'Predictions saved to submit.txt')
