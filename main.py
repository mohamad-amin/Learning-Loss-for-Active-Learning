'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import shutil
import random
import argparse

import utils


# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, SVHN

# Utils
import yaml
import pyaml
from tqdm import tqdm

# Custom
import models.resnet as resnet, models.wide_resnet as wide_resnet
import models.lossnet as lossnet
from data.sampler import SubsetSequentialSampler

# Seed
random.seed("123123123")
torch.manual_seed(12344321)
torch.backends.cudnn.deterministic = True

# Data
DATASET = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'svhn': SVHN
}

TRAIN_TRANSFORM = {
    'cifar100': T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ]),
    'cifar10': T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
    'svhn': T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
}

TEST_TRANSFORM = {
    'cifar100': T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
    'cifar10': T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
    'svhn': T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

}

MODELS = {
    'resnet18': resnet.ResNet18,
    'wideresnet_1bl': wide_resnet.WideResNet1BL,
    'wideresnet_2bl': wide_resnet.WideResNet2BL
}

MODEL_FEATURES = {
    'resnet18': 4,
    'wideresnet_1bl': 1,
    'wideresnet_2bl': 2
}


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


# Train Utils
iters = 0


def train_epoch(models, criterion, optimizers, schedulers,
                dataloaders, epoch, epoch_loss, margin, weight, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            for i in range(models['model_features']):
                features[i] = features[i].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=margin)
        loss = m_backbone_loss + weight * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        # Todo: comment for OneCycleLR
        schedulers['backbone'].step()
        schedulers['module'].step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                m_backbone_loss.item(),
                m_module_loss.item(),
                loss.item()
            ])
            # Todo: replace with log report
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def train(models, criterion, optimizers, schedulers,
          dataloaders, num_epochs, epoch_loss, margin, weight, vis, plot_data):
    best_acc = 0.
    # checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        # Todo: enable for multistep
        # schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch(models, criterion, optimizers, schedulers,
                    dataloaders, epoch, epoch_loss, margin, weight, vis, plot_data)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


# Main
if __name__ == '__main__':

    # Fixing the pthread_cancel glitch while using python 3.8 (you can comment these two lines if you're on 3.7)
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='', help="Path to a config")
    parser.add_argument('--save_dir', default='', help='Path to dir to save checkpoints and logs')
    args = parser.parse_args()

    # utils.set_seeds(args.config_path)

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))
    config = utils.load_config(args.config_path)
    train_config = config['train']
    al_config = config['al']
    data_config = config['data']
    model_config = config['model']

    print(pyaml.dump(config))

    # vis = visdom.Visdom(server='http://localhost', port=9000)
    vis = None
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}


    dataset = DATASET[data_config['name']]
    train_transform = TRAIN_TRANSFORM[data_config['name']]
    test_transform = TEST_TRANSFORM[data_config['name']]

    if data_config['name'] == 'svhn':
        data_train = dataset('../ntk-al/data/SVHN', split='train', download=True, transform=train_transform)
        data_unlabeled = dataset('../ntk-al/data/SVHN', split='train', download=True, transform=test_transform)
        data_test = dataset('../ntk-al/data/SVHN', split='test', download=True, transform=test_transform)
    else:
        data_train = dataset('../ntk-al/data', train=True, download=True, transform=train_transform)
        data_unlabeled = dataset('../ntk-al/data', train=True, download=True, transform=test_transform)
        data_test = dataset('../ntk-al/data', train=False, download=True, transform=test_transform)

    for trial in range(train_config['trials']):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(al_config['num_train']))
        random.shuffle(indices)
        labeled_set = indices[:al_config['init_num']]
        unlabeled_set = indices[al_config['added_num']:]
        
        train_loader = DataLoader(data_train, batch_size=train_config['batch'],
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(data_test, batch_size=train_config['batch'])
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        model_name = model_config['name']
        num_features = MODEL_FEATURES[model_name]
        if 'wide' in model_name:
            widen_factor = model_config['widen_factor']
            feature_sizes = [28, 14, 7][:num_features]
            num_channels = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor][:num_features]
        else:
            feature_sizes = [32, 16, 8, 4]
            num_channels = [64, 128, 256, 512]

        model       = MODELS[model_name](num_classes=10).cuda()
        loss_module = lossnet.LossNet(MODEL_FEATURES[model_name], feature_sizes, num_channels).cuda()
        models      = {'backbone': model, 'module': loss_module, 'model_features': num_features}
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(train_config['cycles']):
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=train_config['lr'],
                                    momentum=train_config['momentum'], weight_decay=train_config['weight_decay'])
            optim_module   = optim.SGD(models['module'].parameters(), lr=train_config['lr'],
                                    momentum=train_config['momentum'], weight_decay=train_config['weight_decay'])

            # sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=train_config['milestones'])
            # sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=train_config['milestones'])

            sched_backbone = lr_scheduler.OneCycleLR(optim_backbone, train_config['lr'],
                                                     epochs=train_config['epoch'], steps_per_epoch=len(train_loader))
            sched_module = lr_scheduler.OneCycleLR(optim_module, train_config['lr'],
                                                     epochs=train_config['epoch'], steps_per_epoch=len(train_loader))

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, train_config['epoch'],
                  train_config['epoch_l'], al_config['margin'], al_config['weight'], vis, plot_data)

            acc = test(models, dataloaders, mode='test')

            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(
                trial+1, train_config['trials'], cycle+1, train_config['cycles'], len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:al_config['subset']]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(data_unlabeled, batch_size=train_config['batch'],
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            if al_config['acquisition'] == 'll4al':
                # Index in ascending order
                arg = np.argsort(uncertainty)
            else:
                arg = np.random.choice(len(uncertainty), len(uncertainty), False)
            
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-al_config['added_num']:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-al_config['added_num']].numpy()) \
                            + unlabeled_set[al_config['subset']:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=train_config['batch'],
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
        
        # # Save a checkpoint
        # torch.save({
        #             'trial': trial + 1,
        #             'state_dict_backbone': models['backbone'].state_dict(),
        #             'state_dict_module': models['module'].state_dict()
        #         },
        #         './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))