import os
import time
import argparse
import numpy as np
import shutil
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from function.models import REGNN, GNN, old_GNN
import pandas as pd
from function.escapev4_dataset import EscapeV4Dataset

from function.utils import EarlyStopping, args_print, set_seed
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

#  参数设置
parser = argparse.ArgumentParser(description='TenCrossTrain')
# 设备与模型类型
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--test_fold', type=int, default=0)
parser.add_argument('--model', type=str, default='regcn', help='regcn, gcn')
# 数据子集与模型结构配置
parser.add_argument('-d', '--subdataset', type=str, default='WT')
parser.add_argument('--num_gnn_layers', type=int, default=3) # GNN 层数
parser.add_argument('--hidden', type=int, default=64) # 隐藏层维度
# 训练参数
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.001) # 学习率
parser.add_argument('--wd', type=float, default=0.) # 权重衰减系数
parser.add_argument('--epochs', type=int, default=200) # 最大训练轮数
parser.add_argument('--early_stop', type=int, default=20) # 提早停止的容忍轮数
parser.add_argument('--fold', type=int, default=5) # 指定运行次数
parser.add_argument('-b', '--train_batch_size', type=int, default=5) # 批量大小
parser.add_argument('-r', '--scaling_factor', type=float, default=1.) # 数据缩放因子
parser.add_argument('--no_re', action='store_true', default=False)
parser.add_argument('--norm', type=str, default='none', help='none, bn, ln') # 指定归一化方法
parser.add_argument('--pooling', type=str, default='readout', help='mean, sum, max, readout, concat') # 池化方法
parser.add_argument('--use_scheduler', action='store_true') # 是否使用学习率调度器
parser.add_argument('--plots_fig', action='store_true') # 是否保存训练图像
parser.add_argument('-l', '--use_linear_label', action='store_true') # 线性标签
parser.add_argument('-m', '--mu2_ratio', type=float, default=0.1)
parser.add_argument('-s', '--split_type', type=int, default=0) # 指定数据集分割类型
parser.add_argument('-c', '--comments', type=str, default='escape_WT') # 指定注释信息
parser.add_argument('--no_test', action='store_true') # 如果设置,则不进行测试

args = parser.parse_args()
args_print(args)

set_seed(123)

# 数据加载与划分
dataset = EscapeV4Dataset(root='traindata/', name=args.subdataset)
in_dim = dataset[0].x.size(1) # 获取输入维度
print(f"数据集样本数: {len(dataset)}")

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

# 模型构建与优化器设置
if args.model == 'regcn':
    model = REGNN(in_dim, args.hidden, 1, args.num_gnn_layers, args.dropout, graph_pooling=args.pooling,
                norm=args.norm, scaling_factor=args.scaling_factor, no_re=args.no_re).to(device)
elif args.model == 'gcn':
    model = GNN(in_dim, args.hidden, 1, args.num_gnn_layers, args.dropout, graph_pooling=args.pooling).to(device)
elif args.model == 'old_gcn':
    model = old_GNN(in_dim, args.hidden, 1, args.num_gnn_layers, args.dropout, graph_pooling=args.pooling).to(device)
else:
    assert False, 'Invalid model.'
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
loss_fn = nn.MSELoss().to(device) # 均方误差损失

# 训练
def train(data_loader):
    model.train() # 将模型设置为训练模式
    outputs, labels = np.array([]), np.array([]) # 初始化两个空的 NumPy 数组,用于存储模型的输出和对应的标签
    loss_all, num_samples = 0, 0 # 初始化总损失和样本数量为 0
    for data in data_loader: # 开始遍历数据加载器中的数据
        data = data.to(device) # 将数据移动到指定的设备(CPU 或 GPU)上
        optimizer.zero_grad() # 将优化器的梯度清零
        output = model(data).reshape(-1) # 将模型的输出进行 reshape,使其变为一维
        loss = loss_fn(output, data.y) * 10 # 计算损失函数,并乘以 10 (可能是为了调整损失函数的数值)
        loss.backward() #  计算梯度
        loss_all += data.num_graphs * loss.item() # 累加总损失
        num_samples += data.num_graphs # 累加样本数量
        optimizer.step()  # 更新参数
        outputs = np.concatenate((outputs, output.cpu().detach().numpy()))
        labels = np.concatenate((labels, data.y.cpu().detach().numpy()))
    if args.use_scheduler:
        scheduler.step(loss_all / num_samples)
    r2 = r2_score(labels, outputs)
    # Spearman's rank correlation
    spearman_corr, _ = spearmanr(labels, outputs)
    # Pearson's correlation coefficient
    pearson_corr, _ = pearsonr(labels, outputs)
    # MAE
    mae = mean_absolute_error(labels, outputs)
    # RMSE
    rmse = np.sqrt(mean_squared_error(labels, outputs))
    return loss_all / num_samples, r2, spearman_corr, pearson_corr, mae, rmse, outputs, labels

# 测试
def test(dataloader):
    model.eval()
    outputs, labels = np.array([]), np.array([])
    loss_all, num_samples = 0, 0
    # with torch.no_grad():
    for data in dataloader:
        data = data.to(device)
        output = model(data).reshape(-1) # 预测结果
        loss = loss_fn(output, data.y)
        loss_all += data.num_graphs * loss.item()
        num_samples += data.num_graphs
        outputs = np.concatenate((outputs, output.cpu().detach().numpy()))
        labels = np.concatenate((labels, data.y.cpu().detach().numpy()))
    r2 = r2_score(labels, outputs)
    # Spearman's rank correlation
    spearman_corr, _ = spearmanr(labels, outputs)
    # Pearson's correlation coefficient
    pearson_corr, _ = pearsonr(labels, outputs)
    # MAE
    mae = mean_absolute_error(labels, outputs)
    # RMSE
    rmse = np.sqrt(mean_squared_error(labels, outputs))
    return loss_all / num_samples, r2, spearman_corr, pearson_corr, mae, rmse, outputs, labels

# 创建十折交叉验证
kf = KFold(n_splits=args.fold, shuffle=True, random_state=123)
train_results = []

# 保存最优模型
best_fold = 0
best_epoch = 0
best_loss = float('inf')
best_r2 = float('inf')
best_mae = float('inf')
best_rmse = float('inf')
best_spearmancorr = float('inf')
best_pearsoncorr = float('inf')
best_model_state = model.state_dict()

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    if args.test_fold != fold + 1:
        continue
    # 保存路径
    save_model_folder = f'train_results/checkpoint/{args.comments}/{fold+1}'
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    save_plots_folder = f'train_results/plots/{args.comments}/{fold+1}'
    shutil.rmtree(save_plots_folder, ignore_errors=True)
    os.makedirs(save_plots_folder, exist_ok=True)

    # 划分训练集和验证集
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    # 初始化模型
    model.reset_parameters()
    best_val_loss = float('inf')
    best_val_r2 = float('inf')
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    best_val_spearmancorr = float('inf')
    best_val_pearsoncorr = float('inf')
    best_val_epoch = 0
    best_val_model_state = model.state_dict()
    es = EarlyStopping(args.early_stop)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_r2, train_spearman_corr, train_pearson_corr, train_mae, train_rmse, train_pred, train_y = train(train_loader)
        val_loss, val_r2, val_spearman_corr, val_pearson_corr, val_mae, val_rmse, val_pred, val_y = test(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_val_r2 = val_r2
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_val_spearmancorr = val_spearman_corr
            best_val_pearsoncorr = val_pearson_corr
            best_val_model_state = model.state_dict()  # 保存最佳模型状态

        if args.plots_fig:
            torch.save(model.state_dict(), f'{save_model_folder}/e{epoch}.pt')

            fig = plt.figure()
            ax = plt.subplot()
            plt.scatter(train_pred, train_y)
            plt.xlabel('Prediction', fontdict={'fontsize': 11})
            plt.ylabel('Ground truth', fontdict={'fontsize': 11})
            plt.title(f'Train Set ({args.subdataset}) [R2 = {train_r2:.2f}]', fontdict={'fontsize': 19})
            plt.savefig(f'{save_plots_folder}/e{epoch}_train.jpg')
            plt.close(fig)

            fig = plt.figure()
            ax = plt.subplot()
            plt.scatter(val_pred, val_y)
            plt.xlabel('Prediction', fontdict={'fontsize': 11})
            plt.ylabel('Ground truth', fontdict={'fontsize': 11})
            plt.title(f'Val Set ({args.subdataset}) [R2 = {val_r2:.2f}]', fontdict={'fontsize': 19})
            plt.savefig(f'{save_plots_folder}/e{epoch}_val.jpg')
            plt.close(fig)

        es(val_r2)
        if es.early_stop:
            break
        if args.use_scheduler:
            print('Fold: {:03d}, Epoch: {:03d}, Train Loss: {:.4f}, Spearman R: {:.4f}, MAE: {:.4f}, Val Loss: {:.4f}, Spearman R: {:.4f}, MAE: {:.4f}, lr: {:.8f}'.
                  format(fold+1, epoch, train_loss, train_spearman_corr, train_mae, val_loss, val_spearman_corr, val_mae, optimizer.param_groups[0]['lr']))
        else:
            print('Fold: {:03d}, Epoch: {:03d}, Train Loss: {:.4f}, Spearman R: {:.4f}, MAE: {:.4f}, Val Loss: {:.4f}, Spearman R: {:.4f}, MAE: {:.4f}'.
                  format(fold+1, epoch, train_loss, train_spearman_corr, train_mae, val_loss, val_spearman_corr, val_mae))

    # 保存每折的最佳结果
    train_results.append((fold + 1, best_val_epoch, best_val_loss, best_val_mae, best_val_rmse, best_val_spearmancorr, best_val_pearsoncorr, best_val_r2))
    print((fold + 1, best_val_epoch, best_val_loss, best_val_mae, best_val_rmse, best_val_spearmancorr, best_val_pearsoncorr, best_val_r2))

    print(f"Fold {fold + 1} completed: Best Val Loss = {best_val_loss:.4f} at Epoch {best_epoch}")

    if best_val_loss < best_loss:
        best_loss = best_val_loss
        best_fold = fold
        best_epoch = best_val_epoch
        best_r2 = best_val_r2
        best_mae = best_val_mae
        best_rmse = best_val_rmse
        best_spearmancorr = best_val_spearmancorr
        best_pearsoncorr = best_val_pearsoncorr
        best_model_state = best_val_model_state  # 保存最佳模型状态

print('Fold: {:03d}, Epoch: {:03d}, Loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, Spearman R: {:.4f}, Pearson R: {:.4f}, R2: {:4f}'.
                  format(best_fold+1, best_epoch, best_loss, best_mae, best_rmse, best_spearmancorr, best_pearsoncorr,best_r2))
exit()
print(f"Final Model completed: Fold {best_fold + 1}, Epoch {best_epoch}, Best Loss = {best_loss:.4f}, Best R² = {best_r2:.4f}, Best MAE = {best_mae:.4f}")
# 保存最终模型
torch.save(best_model_state, f"models/{args.comments}.pt")
print(f"Final model saved as '{args.comments}.pt'")

train_results.append((f'save {best_fold + 1}', best_epoch, best_loss, best_mae, best_rmse, best_spearmancorr, best_pearsoncorr, best_r2))
# 定义列名和创建 DataFrame
columns = ['Fold', 'Best Epoch', 'Best Val Loss', 'Best Val MAE', 'Best Val RMSE', 'Best Val SpearmanR', 'Best Val PearsonR', 'Best Val R2']
df_results = pd.DataFrame(train_results, columns=columns)

# 保存为 CSV 文件
output_file = f'train_results/{args.comments}.csv'
df_results.to_csv(output_file, index=False)

print(f"Train results saved to {output_file}")
