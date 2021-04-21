import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

# load model from pytorch
net = models.resnet18(pretrained=True)
net.avgpool =nn.AdaptiveAvgPool2d((1, 1))
net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
# print(net)
net = net.cuda()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()])
train_dataset = ImageFolder(root='./face/', transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

optim = torch.optim.Adam(net.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()
summary = SummaryWriter()

num_epochs = 20
iter = 0
# training
for epoch in range(1, 1+num_epochs):
    train_bar = tqdm(train_data_loader)
    running_results = {'batch_sizes': 0, 'loss': 0, 'error_num': 0, 'total_num': 0}
    for data, label in train_bar:
        iter += 1
        running_results['batch_sizes'] += 1

        data = data.cuda()
        label = label.cuda()

        out = net(data)
        loss = loss(out, label.long())
        net.zero_grad()
        loss.backward()
        optim.step()
        pre_label_onehot = F.log_softmax(out, 1) #logsoftmax
        topk_values, topk_indices = pre_label_onehot.topk(1, dim=1)

        error_num = 0
        for i, item in enumerate(label):
            if item not in topk_indices[i]:
                error_num += 1

        running_results['error_num'] += error_num
        running_results['total_num'] += data.shape[0]

        running_results['loss'] += loss.item()
        train_bar.set_description(desc='[%d/%d] Loss: %.4f  TOP1-ACC: %.4f' %
           (epoch, num_epochs, running_results['loss'] / running_results['batch_sizes'],
            1 - (running_results['error_num'] / running_results['total_num']))
        )
        summary.add_scalar('loss', loss, iter)
        summary.add_scalar('top1-acc', 1 - (error_num / data.shape[0]), iter)

        # save model
        if epoch % 1 == 0:
           torch.save(net.state_dict(), './epochs/epoch_%d.pth'%epoch)

