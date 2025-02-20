import os
import time
import argparse
import numpy as np
import shutil
from sklearn.metrics import r2_score

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from models1 import REGNN, GNN, old_GNN

from geno_dataset import GENODataset
from genofull_dataset import GENOFULLDataset
from utils import EarlyStopping, args_print, set_seed
import matplotlib.pyplot as plt
import wandb


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default='regcn', help='regcn, gcn')
parser.add_argument('--num_gnn_layers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('-b', '--train_batch_size', type=int, default=5)
parser.add_argument('-r', '--scaling_factor', type=float, default=10.)
# parser.add_argument('--residual', action='store_true', default=False)
parser.add_argument('--no_re', action='store_true', default=False)
parser.add_argument('--norm', type=str, default='none', help='none, bn, ln')
parser.add_argument('--pooling', type=str, default='max', help='mean,sum, max, readout')
parser.add_argument('--use_scheduler', action='store_true')
parser.add_argument('--plots_fig', action='store_true')
parser.add_argument('-l', '--use_linear_label', action='store_true')
parser.add_argument('-m', '--mu2_ratio', type=float, default=0.2)
parser.add_argument('-c', '--comments', type=str, default='raw')

args = parser.parse_args()
args_print(args)

wandb.tensorboard.patch(root_logdir='logs/'+args.comments)
wandb.init(
      project="Geno_Mu1to2", 
      name=args.comments, 
      config=args,
      sync_tensorboard=True)

set_seed(123)

train_val_dataset = GENODataset(root='./data/', name='all')
train_val_dataset = train_val_dataset.shuffle()
print(train_val_dataset)
test_dataset = GENODataset(root='./data/', name='all', mut=2, linear_label=args.use_linear_label)
test_dataset = test_dataset.shuffle()

in_dim = train_val_dataset[0].x.size(1)

# 20% training samples are used for validation
num_test_0 = int(len(test_dataset) * args.mu2_ratio)
print(num_test_0)
test_dataset_0 = test_dataset[:num_test_0]
test_dataset_1 = test_dataset[num_test_0:]
print(test_dataset_0)
print(test_dataset_1)
num_test_0_val = len(test_dataset_0) // 5
test_0_val_dataset = test_dataset_0[:num_test_0_val]
test_0_train_dataset = test_dataset_0[num_test_0_val:]

# 20% training samples are used for validation
num_val = len(train_val_dataset) // 5
val_dataset = train_val_dataset[:num_val] + test_0_val_dataset
train_dataset = train_val_dataset[num_val:] + test_0_train_dataset
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_1, batch_size=1000, shuffle=False)

if args.model == 'regcn':
    model = REGNN(in_dim, args.hidden, 1, args.num_gnn_layers, args.dropout, graph_pooling=args.pooling, 
                norm=args.norm, scaling_factor=args.scaling_factor, no_re=args.no_re).to(device)
elif args.model == 'gcn':
    model = GNN(in_dim, args.hidden, 1, args.num_gnn_layers, args.dropout, graph_pooling=args.pooling).to(device)
elif args.model == 'old_gcn':
    model = old_GNN(in_dim, args.hidden, 1, args.num_gnn_layers, args.dropout, graph_pooling=args.pooling).to(device)
else:
    assert False, 'Invalid model.'
# model.load_state_dict(torch.load(f'checkpoint/e25_gcn.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
loss_fn = nn.MSELoss().to(device)
# loss_fn = nn.HuberLoss().to(device)

def train(data_loader):
    model.train()
    outputs, labels = np.array([]), np.array([])
    loss_all, num_samples = 0, 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).reshape(-1)
        loss = loss_fn(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        num_samples += data.num_graphs
        optimizer.step()
        outputs = np.concatenate((outputs, output.cpu().detach().numpy()))
        labels = np.concatenate((labels, data.y.cpu().detach().numpy()))
    if args.use_scheduler:
        scheduler.step(loss_all / num_samples)
    r2 = r2_score(labels, outputs)

    return loss_all / num_samples, r2


def test(dataloader):
    model.eval()
    outputs, labels = np.array([]), np.array([])
    loss_all, num_samples = 0, 0
    # with torch.no_grad():
    for data in dataloader:
        data = data.to(device)
        output = model(data).reshape(-1)
        loss = loss_fn(output, data.y)
        loss_all += data.num_graphs * loss.item()
        num_samples += data.num_graphs
        outputs = np.concatenate((outputs, output.cpu().detach().numpy()))
        labels = np.concatenate((labels, data.y.cpu().detach().numpy()))
    r2 = r2_score(labels, outputs)
    return loss_all / num_samples, r2, outputs, labels


test_r2s = []
max_test_r2s = []
durations = []
for run in range(1, args.runs + 1):
    st = time.perf_counter()
    best_val_loss = 1e10
    best_val_r2, best_test_r2 = 0., 0.
    max_test_r2 = 0.0
    es = EarlyStopping(args.early_stop)
    model.reset_parameters()

    logs_folder = f'logs/{args.comments}'
    shutil.rmtree(logs_folder, ignore_errors=True)
    os.makedirs(logs_folder, exist_ok=True)
    save_model_folder = f'checkpoint/{args.comments}'
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    save_plots_folder = f'plots/{args.comments}'
    shutil.rmtree(save_plots_folder, ignore_errors=True)
    os.makedirs(save_plots_folder, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss, r2 = train(train_loader)
        train_loss, train_r2, train_pred, train_y = test(train_loader)
        val_loss, val_r2, val_pred, val_y = test(val_loader)
        test_loss, test_r2, test_pred, test_y = test(test_loader)
        wandb.log({'r2/1train': r2, 'r2/val': val_r2, 'r2/2test': test_r2,
            'loss/1train': loss, 'loss/val': val_loss, 'loss/2test': test_loss})
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_test_r2 = test_r2
            wandb.run.summary["best_test_r2"] = best_test_r2
        if max_test_r2 < test_r2:
            max_test_r2 = test_r2
            wandb.run.summary["max_test_r2"] = max_test_r2
        if args.plots_fig:
            # save model dict
            torch.save(model.state_dict(), f'{save_model_folder}/e{epoch}.pt')

            fig = plt.figure()
            ax = plt.subplot()
            plt.scatter(test_pred, test_y)
            plt.xlim((0.5, 11.5))
            plt.ylim((0.5, 11.5))
            plt.xlabel('Prediction',fontdict={'fontsize':11})
            plt.ylabel('Ground truth',fontdict={'fontsize':11})
            plt.title(f'Test Set (BA2) [R2 = {test_r2:.2f}]',fontdict={'fontsize':19})
            plt.show()
            plt.savefig(f'{save_plots_folder}/e{epoch}_test.jpg')

            fig = plt.figure()
            ax = plt.subplot()
            plt.scatter(train_pred, train_y)
            plt.xlim((0.5, 11.5))
            plt.ylim((0.5, 11.5))
            plt.xlabel('Prediction',fontdict={'fontsize':11})
            plt.ylabel('Ground truth',fontdict={'fontsize':11})
            plt.title(f'Train Set [R2 = {train_r2:.2f}]',fontdict={'fontsize':19})
            plt.show()
            plt.savefig(f'{save_plots_folder}/e{epoch}_train.jpg')

            fig = plt.figure()
            ax = plt.subplot()
            plt.scatter(val_pred, val_y)
            plt.xlim((0.5, 11.5))
            plt.ylim((0.5, 11.5))
            plt.xlabel('Prediction',fontdict={'fontsize':11})
            plt.ylabel('Ground truth',fontdict={'fontsize':11})
            plt.title(f'Validation Set [R2 = {val_r2:.2f}]',fontdict={'fontsize':19})
            plt.show()
            plt.savefig(f'{save_plots_folder}/e{epoch}_val.jpg')
            
            # print("save figure!")
        es(val_r2)
        if es.early_stop:
            break
        if args.use_scheduler:
            print('Epoch: {:03d}, Loss: {:.4f}, r2: {:.4f}, Val Loss: {:.4f}, r2: {:.4f}, Test r2: {:.4f}, lr: {:.8f}'.
                format(epoch, loss, r2, val_loss, val_r2, test_r2, optimizer.param_groups[0]['lr']))
        else:
            print('Epoch: {:03d}, Loss: {:.4f}, r2: {:.4f}, Val Loss: {:.4f}, r2: {:.4f}, Test r2: {:.4f}'.
                format(epoch, loss, r2, val_loss, val_r2, test_r2))
    et = time.perf_counter()
    durations.append(et - st)
    test_r2s.append(best_test_r2)
    max_test_r2s.append(max_test_r2)
    print('Run: {:02d}, Test R2: {:.4f}, Max Test R2: {:.4f}, Time: {:.2f}'.format(run, best_test_r2, max_test_r2, et-st))
    print()


test_r2s, durations = np.array(test_r2s), np.array(durations)
max_test_r2s = np.array(max_test_r2s)
print("Test R2s", test_r2s)
print("Max Test R2s", max_test_r2s)
print("Avg Test R2: {:.4f}, Std: {:.4f}, Time: {:.2f}".format(test_r2s.mean(), test_r2s.std(), durations.mean()))
print(args.comments)
