from model import *
import os
import numpy as np
import torch
import torch.nn as nn
from losses import *
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from dataset import Dataset
from model import U_Net
from torch.optim.lr_scheduler import MultiStepLR

batch_size = 4
ROOT = r'/yy/data/seg_right'
output_dir = r'/yy/code/pix2pix/output/checkpoint/seg_right'
epochs = 50
train_dataset = Dataset('train',ROOT)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
model = U_Net()
#损失函数
criterion = nn.BCEWithLogitsLoss()


model.cuda()

model.train()
optimizer = optim.Adam(model.parameters(),lr = 0.01,betas = (0.5,0.999))
scheduler = MultiStepLR(optimizer, [30, 55, 80], gamma=0.5)


best_loss = 999999
for epoch in range(epochs):
    print('Epoch [%d/%d]' % (epoch + 1, epochs))
    train_losses = []
    for data in train_loader:
        model.zero_grad()
        inp = data['inp']
        label = data['gt']
        inp = inp.cuda()
        label = label.cuda()
        output = model(inp)
        #print(label.shape)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    loss_mean = sum(train_losses)/len(train_losses)
    if loss_mean<best_loss:
        best_loss = loss_mean
        torch.save(model.state_dict(),os.path.join(output_dir,'best_segement.pth'))

    scheduler.step()
    print("epoch-{};loss:{:.4}".format(epoch+1,loss_mean))






