from sklearn.metrics import r2_score
import numpy as np
import torch
import sys
sys.path.append('../../geno_gnn')
from models import REGNN, GNN, old_GNN
from geno_dataset import GENODataset_update
from torch_geometric.loader import DataLoader
import pandas as pd

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

loss_fn = torch.nn.MSELoss().to(device)

def test(dataloader,model,loss_fn, savepth):
    model.eval()
    outputs, labels = np.array([]), np.array([])
    embs = None
    loss_all, num_samples = 0, 0
    # with torch.no_grad():
    for data in dataloader:
        data = data.to(device)
        output, emb = model(data, return_emb=True)
        output = output.reshape(-1)
        loss = loss_fn(output, data.y)
        loss_all += data.num_graphs * loss.item()
        num_samples += data.num_graphs
        outputs = np.concatenate((outputs, output.cpu().detach().numpy()))
        if embs is None:
            embs = emb.cpu().detach().numpy()
        else:
            s = np.concatenate((embs, emb.cpu().detach().numpy()))

        labels = np.concatenate((labels, data.y.cpu().detach().numpy()))
    print(np.any(np.isnan(labels)), np.any(np.isnan(outputs)))
    r2 = r2_score(outputs, labels)
    corr = np.corrcoef(labels,outputs)
    print(corr)
    pd.DataFrame({'labels':labels,'outputs':outputs}).to_csv(savepth, index=0)
    return loss_all / num_samples, r2, outputs, embs, labels


model = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                  norm='none', scaling_factor=10., no_re=False).to(device)
dataset = GENODataset_update(root='testdata',name='affinity')
loader = DataLoader(dataset, batch_size=5000, shuffle=False)
model.load_state_dict(torch.load(f'models/affinity.pt'))
loss, r2, pred, emb, y = test(loader, model, loss_fn,'test_results/affinity_ba1_test.csv')

dataset = GENODataset_update(root='testdata',name='affinity_Omicron')
loader = DataLoader(dataset, batch_size=5000, shuffle=False)
model.load_state_dict(torch.load(f'models/affinity.pt'))
loss, r2, pred, emb, y = test(loader, model, loss_fn,'test_results/affinity_omicron_test.csv')

dataset = GENODataset_update(root='testdata/escape/data',name='CoronaVac')
loader = DataLoader(dataset, batch_size=5000, shuffle=False)
model1 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                   norm='none', scaling_factor=1., no_re=False).to(device)
model1.load_state_dict(torch.load(f'models/escape_VAC.pt'))
loss, r2, pred, emb, y = test(loader,model1,loss_fn,'test_results/escape_CoronaVac_test.csv')

dataset = GENODataset_update(root='testdata/escape/data',name='BA1')
loader = DataLoader(dataset, batch_size=5000, shuffle=False)
model2 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                   norm='none', scaling_factor=1., no_re=False).to(device)
model2.load_state_dict(torch.load(f'models/escape_BA1.pt'))
loss, r2, pred, emb, y = test(loader,model2,loss_fn,'test_results/escape_BA1_test.csv')

dataset = GENODataset_update(root='testdata/escape/data',name='BA2')
loader = DataLoader(dataset, batch_size=5000, shuffle=False)
model3 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                   norm='none', scaling_factor=1., no_re=False).to(device)
model3.load_state_dict(torch.load(f'models/escape_BA2.pt'))
loss, r2, pred, emb, y = test(loader,model3,loss_fn,'test_results/escape_BA2_test.csv')

dataset = GENODataset_update(root='testdata/escape/data',name='BA5')
loader = DataLoader(dataset, batch_size=5000, shuffle=False)
model4 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                   norm='none', scaling_factor=1., no_re=False).to(device)
model4.load_state_dict(torch.load(f'models/escape_BA5.pt'))
loss, r2, pred, emb, y = test(loader,model4,loss_fn,'test_results/escape_BA5_test.csv')