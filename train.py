from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from data_utils import PDBBindDataset, accuracy
from models import Model
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = PDBBindDataset(num_positive=args.num_positive,
                         num_negative=args.num_negative)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(args.train_test_split * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices, val_indices \
    = indices[2*split:], indices[:split], indices[split:2*split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, 
                          #batch_size=args.batch_size,
                          batch_size=1, # Loads graph one at a time
                          sampler=train_sampler)
test_loader = DataLoader(dataset, 
                         #batch_size=args.batch_size,
                         batch_size=1,
                         sampler=test_sampler)
val_loader = DataLoader(dataset, 
                        #batch_size=args.batch_size,
                        batch_size=1,
                        sampler=val_sampler)

model = Model(n_out=dataset.__nlabels__(),
              n_feat=dataset.__nfeats__(), 
              n_attns=args.n_attns, 
              n_dense=args.n_dense,
              dim_attn=args.dim_attn,
              dim_dense=args.dim_dense,
              dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    #features = features.cuda()
    #adj = adj.cuda()
    #labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    for feature, label in train_loader:
        X, A, D = feature

        if args.cuda:
            X = X.cuda()
            A = A.cuda()
            D = D.cuda()
            label = label.cuda()

        output = model(X=X.squeeze(), 
                       A=A.squeeze(), 
                       D=D.squeeze())
        loss_train = F.nll_loss(output.unsqueeze(0), label.long())
        acc_train = accuracy(output, label)
        print(loss_train)
        print(acc_train)
        loss_train.backward()
        optimizer.step()
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          #'loss_val: {:.4f}'.format(loss_val.data.item()),
          #'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    return loss_val.data.item()


def compute_test():
    model.eval()

    for feature, label in test_loader:
        X, A, D = feature

        if args.cuda:
            X = X.cuda()
            A = A.cuda()
            D = D.cuda()
            label = label.cuda()

        output = model(X=X.squeeze(), 
                       A=A.squeeze(), 
                       D=D.squeeze())
        loss_test = F.nll_loss(output.unsqueeze(0), label.long())
        acc_test = accuracy(output, label)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    try:
        loss_values.append(train(epoch))

        if epoch % 20 == 0:
            model.train()
            for feature, label in val_loader:
                X, A, D = feature

                if args.cuda:
                    X = X.cuda()
                    A = A.cuda()
                    D = D.cuda()
                    label = label.cuda()

                output = model(X=X.squeeze(), 
                               A=A.squeeze(), 
                               D=D.squeeze())
                loss_val = F.nll_loss(output.unsqueeze(0), label.long())
                acc_val = accuracy(output, label)

            print('Epoch: {:04d}'.format(epoch+1),
                  #'loss_train: {:.4f}'.format(loss_train.data.item()),
                  #'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  #'time: {:.4f}s'.format(time.time() - t)
                  )
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    except:
        pass

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()