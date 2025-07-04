import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple
import re
import torch
import torch.nn as nn
import time
import torch.optim as optim

# tokenizer，鉴于SMILES的特性，这里需要自己定义tokenizer和vocab
# 这里直接将smiles str按字符拆分，并替换为词汇表中的序号
class Smiles_tokenizer():
    def __init__(self, pad_token, regex, vocab_file, max_length):
        self.pad_token = pad_token
        self.regex = regex
        self.vocab_file = vocab_file
        self.max_length = max_length

        with open(self.vocab_file, "r") as f:
            lines = f.readlines()
        lines = [line.strip("\n") for line in lines]
        vocab_dic = {}
        for index, token in enumerate(lines):
            vocab_dic[token] = index
        self.vocab_dic = vocab_dic

    def _regex_match(self, smiles):
        regex_string = r"(" + self.regex + r"|"
        regex_string += r".)"
        prog = re.compile(regex_string)

        tokenised = []
        for smi in smiles:
            tokens = prog.findall(smi)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            tokenised.append(tokens)  # 返回一个所有的字符串列表
        return tokenised

    def tokenize(self, smiles):
        tokens = self._regex_match(smiles)
        # 添加上表示开始和结束的token：<cls>, <end>
        tokens = [["<CLS>"] + token + ["<SEP>"] for token in tokens]
        tokens = self._pad_seqs(tokens, self.pad_token)
        token_idx = self._pad_token_to_idx(tokens)
        return tokens, token_idx

    def _pad_seqs(self, seqs, pad_token):
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        return padded

    def _pad_token_to_idx(self, tokens):
        idx_list = []
        new_vocab = []
        for token in tokens:
            tokens_idx = []
            for i in token:
                if i in self.vocab_dic.keys():
                    tokens_idx.append(self.vocab_dic[i])
                else:
                    new_vocab.append(i)
                    self.vocab_dic[i] = max(self.vocab_dic.values()) + 1
                    tokens_idx.append(self.vocab_dic[i])
            idx_list.append(tokens_idx)

        with open("../new_vocab_list.txt", "a") as f:
            for i in new_vocab:
                f.write(i)
                f.write("\n")

        return idx_list

    def _save_vocab(self, vocab_path):
        with open(vocab_path, "w") as f:
            for i in self.vocab_dic.keys():
                f.write(i)
                f.write("\n")
        print("update new vocab!")


# 处理数据

def read_data(file_path, train=True):
    df = pd.read_csv(file_path)
    reactant1 = df["Reactant1"].tolist()
    reactant2 = df["Reactant2"].tolist()
    product = df["Product"].tolist()
    additive = df["Additive"].tolist()
    solvent = df["Solvent"].tolist()
    if train:
        react_yield = df["Yield"].tolist()
    else:
        react_yield = [0 for i in range(len(reactant1))]

    # 将reactant\additive\solvent拼到一起，之间用.分开。product也拼到一起，用>>分开
    input_data_list = []
    for react1, react2, prod, addi, sol in zip(reactant1, reactant2, product, additive, solvent):
        # input_info = ".".join([react1, react2, addi, sol])
        input_info = ".".join([react1, react2])
        input_info = ">".join([input_info, prod])
        input_data_list.append(input_info)
    output = [(react, y) for react, y in zip(input_data_list, react_yield)]

    return output


# 定义数据集
class ReactionDataset(Dataset):
    def __init__(self, data: List[Tuple[List[str], float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    REGEX = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    tokenizer = Smiles_tokenizer("<PAD>", REGEX, "../vocab_full.txt", 300)
    smi_list = []
    yield_list = []
    for i in batch:
        smi_list.append(i[0])
        yield_list.append(i[1])
    tokenizer_batch = torch.tensor(tokenizer.tokenize(smi_list)[1])
    yield_list = torch.tensor(yield_list)
    return tokenizer_batch, yield_list


# 模型
# 这里直接采用一个transformer encoder model
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, fnn_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.layerNorm = nn.LayerNorm(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=num_heads,
                                                        dim_feedforward=fnn_dim,
                                                        dropout=dropout,
                                                        batch_first=True,
                                                        norm_first=True  # pre-layernorm
                                                        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_layers,
                                                         norm=self.layerNorm)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Sequential(nn.Linear(d_model, 256),
                                nn.Sigmoid(),
                                nn.Linear(256, 96),
                                nn.Sigmoid(),
                                nn.Linear(96, 1))

    def forward(self, src):
        # src shape: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [batch_size, src_len, d_model]
        outputs = self.transformer_encoder(embedded)
        # outputs shape: [batch_size, src_len, d_model]

        # first
        z = outputs[:, 0, :]
        # z = torch.sum(outputs, dim=1)
        # print(z)
        # z shape: [bs, d_model]
        outputs = self.lc(z)
        # print(outputs)
        # outputs shape: [bs, 1]
        return outputs.squeeze(-1)

def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 训练
def train():
    ## super param
    N = 10  # int / int(len(dataset) * 1)  # 或者你可以设置为数据集大小的一定比例，如 int(len(dataset) * 0.1)
    INPUT_DIM = 292  # src length
    D_MODEL = 512
    NUM_HEADS = 4
    FNN_DIM = 1024
    NUM_LAYERS = 4
    DROPOUT = 0.2
    CLIP = 1  # CLIP value
    N_EPOCHS = 40
    LR = 1e-4

    start_time = time.time()  # 开始计时
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    data = read_data("../dataset/round1_train_data.csv")
    dataset = ReactionDataset(data)
    subset_indices = list(range(N))
    subset_dataset = Subset(dataset, subset_indices)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    model = TransformerEncoderModel(INPUT_DIM, D_MODEL, NUM_HEADS, FNN_DIM, NUM_LAYERS, DROPOUT)
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.MSELoss()

    best_valid_loss = 10
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        # adjust_learning_rate(optimizer, epoch, LR) # 动态调整学习率
        for i, (src, y) in enumerate(train_loader):
            src, y = src.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            epoch_loss += loss.detach().item()

            if i % 50 == 0:
                print(f'Step: {i} | Train Loss: {epoch_loss:.4f}')

        loss_in_a_epoch = epoch_loss / len(train_loader)
        scheduler.step(loss_in_a_epoch)
        print(f'Epoch: {epoch + 1:02} | Train Loss: {loss_in_a_epoch:.3f}')
        if loss_in_a_epoch < best_valid_loss:
            best_valid_loss = loss_in_a_epoch
            # 在训练循环结束后保存模型
            torch.save(model.state_dict(), '../model/transformer.pth')
    end_time = time.time()  # 结束计时
    # 计算并打印运行时间
    elapsed_time_minute = (end_time - start_time) / 60
    print(f"Total running time: {elapsed_time_minute:.2f} minutes")

if __name__ == '__main__':
    train()

# 生成结果文件
def predicit_and_make_submit_file(model_file, output_file):
    INPUT_DIM = 292  # src length
    D_MODEL = 512
    NUM_HEADS = 4
    FNN_DIM = 1024
    NUM_LAYERS = 4
    DROPOUT = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = read_data("../dataset/round1_test_data.csv", train=False)
    test_dataset = ReactionDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = TransformerEncoderModel(INPUT_DIM, D_MODEL, NUM_HEADS, FNN_DIM, NUM_LAYERS, DROPOUT).to(device)
    # 加载最佳模型
    model.load_state_dict(torch.load(model_file))
    model.eval()
    output_list = []
    for i, (src, y) in enumerate(test_loader):
        src = src.to(device)
        with torch.no_grad():
            output = model(src)
            output_list += output.detach().tolist()
    ans_str_lst = ['rxnid,Yield']
    for idx, y in enumerate(output_list):
        ans_str_lst.append(f'test{idx + 1},{y:.4f}')
    with open(output_file, 'w') as fw:
        fw.writelines('\n'.join(ans_str_lst))

predicit_and_make_submit_file("../model/transformer.pth",
                              "../output/result.txt")

