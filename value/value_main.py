import numpy as np
import pandas as pd
# from sklearn.metrics import mean_absolute_error, precision_score, recall_score

# def calculate_metrics(y_true, y_pred, threshold=30):
#     # 计算平均绝对误差
#     mae = np.mean(np.abs(y_true - y_pred))
#
#     # 将实际值和预测值转换为二进制分类（低于阈值为1，高于或等于阈值为0)
#     y_true_binary = (y_true < threshold).astype(int)
#     y_pred_binary = (y_pred < threshold).astype(int)
#
#     # 创建掩码，用于筛选预测值在0和阈值之间的样本
#     mask = (y_pred >= 0) & (y_pred <= threshold)
#     range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100
#
#     # 计算精确度、召回率和F1得分
#     precision = precision_score(y_true_binary, y_pred_binary, average='binary')
#     recall = recall_score(y_true_binary, y_pred_binary, average='binary')
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#
#     # 计算综合评分
#     score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
#
#     return score

def calculate_r2(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score

def main(standard_file, predicted_file):
    standard_data = pd.read_csv(standard_file)
    predicted_data = pd.read_csv(predicted_file)

    y_true = standard_data['Yield'].values
    y_pred = predicted_data['Yield'].values

    # score = calculate_metrics(y_true, y_pred)
    score = calculate_r2(y_true, y_pred)

    print(f'综合评分: {score:.4f}')

# 示例用法
if __name__ == "__main__":
    standard_file = 'test_for_value.csv'
    predicted_file = 'submit_LI_1.txt'
    main(standard_file, predicted_file)
