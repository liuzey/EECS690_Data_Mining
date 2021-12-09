import numpy as np
import os
import copy
import cv2
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
import training
from trojanzoo.utils.output import ansi, output_iter, prints
from trojanzoo import to_list
import torch.nn.utils.prune as prune
import torch.utils.data
from data import Dataset
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torch.nn import functional as F


torch.cuda.empty_cache()
torch.random.manual_seed(100)
prune_ratio = 0.7
method = 'gradcam++'  # Can be gradcam/gradcam++/scorecam
dataset_name = 'kaggle'
if dataset_name == 'LFW':
    ACC = '9991'
elif dataset_name == 'kaggle':
    ACC = 'saved_78'
elif dataset_name == 'realworld':
    ACC = '98'
else:
    raise NotImplementedError
# dataset_name = 'kaggle'
# dataset_name = 'realworld'
pool = nn.AdaptiveAvgPool2d((1, 1))
flatten = nn.Flatten()
BATCH_SIZE = 4
valid_data_dir = './'+dataset_name+'_cropped'
val_dataset = Dataset(dataset_name, valid_data_dir, shuffle_pairs=(dataset_name=='realworld'), augment=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# data_dir = './kaggle/img'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=8631)

        self.cls_head = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(8631, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def model_f(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        f_conv2 = x.clone()
        x = self.model.block8(x)
        f_conv = x.clone()
        x = self.model.avgpool_1a(x)
        x = self.model.dropout(x)
        x = self.model.last_linear(x.view(x.shape[0], -1))
        x = self.model.last_bn(x)
        if self.model.classify:
            x = self.model.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return f_conv, f_conv2, x

    def forward(self, img1, img2):
        feat2 = self.model(img2)
        feat1 = self.model(img1)
        combined_features = feat1 * feat2
        # print(combined_features.shape)
        output = self.cls_head(combined_features)
        return output


model = SiameseNetwork()
resnet = model.to(device)
resnet.load_state_dict(torch.load('./'+dataset_name+'_best_model_'+ACC+'.pth'), strict=True)
resnet.eval()
loss_fn = torch.nn.BCELoss()
metrics = {
    'fps': training.BatchTimer(),
    'TP': training.TP,
    'TP_and_FP': training.TP_and_FP,
    'TP_and_FN': training.TP_and_FN,
    'acc': training.accuracy
}
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10


def test():
    resnet.eval()
    _, metrics_ = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    acc = metrics_['acc'].numpy()
    return 100 * acc


def prune_step(mask, prune_num, index):
    with torch.no_grad():
        feats_list = []
        resnet.eval()
        for (img1, img2), y, (class1, class2) in val_loader:
            img1, img2, y = img1.to(device), img2.to(device), y.to(device)
            f_conv1, f_conv2, class_output = resnet.model_f(img1)
            _feats_list = [f_conv1, f_conv2]
            _feats = _feats_list[index]
            feats_list.append(flatten(pool(_feats)))
        feats_list = torch.cat(feats_list).mean(dim=0)
        idx_rank = to_list(feats_list.argsort())
        # print(idx_rank)
    counter = 0
    for idx in idx_rank:
        if mask[idx].norm(p=1) > 1e-6:
            mask[idx] = 0.0
            counter += 1
            print(f'    {output_iter(counter, prune_num)} Prune {idx:4d} / {len(idx_rank):4d}')
            if counter >= min(prune_num, len(idx_rank)):
                break


mode = 'single'  # In single mode, ablation study is done. In combined mode, pruning is done layer by layer.#     print(item)
prune_list = [resnet.model.block8.conv2d, resnet.model.repeat_3[-1].conv2d]
_accu = test()
print('Accuracy before pruning {:.4f}%'.format(_accu))

if mode == 'combined':
    start, end = 0, 2  # recursively through each layer
else:
    start = 0  # assign layer number here
    end = start + 1

nums = [0, 0]
for k in range(start, end):
    layer = prune_list[k]
    target = prune.identity(layer, 'weight')
    channels = target.out_channels
    print('Total channel count: ', channels)
    prune_num = int(channels * prune_ratio)
    mask = target.weight_mask
    print('Mask shape: ', mask.shape)

    for i in range(prune_num):
        print('\nIter: ', output_iter(i + 1, 10))
        prune_step(mask, prune_num=1, index=k)
        nums[k] += 1
        if i > 142:
            accu = test()
            print('Normal Test accuracy after pruning {}th layer {} neurons {:.4f}%'.format(k, nums[k], accu))
        if nums[k] % 10 == 1:
            # torch.save(resnet.state_dict(), './pruned_models/pruned_{}_neurons.pth'.format(sum(nums)))
            pass
        if mode == 'combined' and _accu - accu > 10:
            break