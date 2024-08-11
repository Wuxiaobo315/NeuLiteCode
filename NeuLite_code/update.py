#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
from CKA import linear_CKA
from HSIC import hsic_normalized_cca,hsic_normalized
import torch.nn.functional as F


kl_loss = nn.KLDivLoss(reduction="batchmean").to('cuda')

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

import torchvision.transforms as transforms
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger,teacher_model,stage,epoch):
        self.args = args
        self.logger = logger
        self.teacher_model = teacher_model
        self.testloader = None
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.idx = idxs



        self.trainloader, self.validloader, self.testloader,self.sorted_idx = self.train_val_test(
            dataset, list(idxs),stage,epoch)
        # Default criterion set to NLL loss function


    def train_val_test(self, dataset, idxs,stage,epoch):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[:int(1*len(idxs))]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=128, shuffle=False)
        self.testloader = testloader
        # if flag==1:
        #     individual_loss = self.inference(self.teacher_model)
        #     sorted_tuples = sorted(zip(individual_loss, idxs_test))
        #     sorted_losses, sorted_idx = zip(*sorted_tuples)
        #     idxs_train = sorted_idx[:int(0.75*len(sorted_idx))]
        #     trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, shuffle=True)
        #     return trainloader, validloader, testloader,sorted_idx
        return trainloader, validloader, testloader, idxs_test

    def update_weights(self, model, global_round,idx,stage,tmp_model):

        model.train()
        tmp_model.eval()
        epoch_loss = []
        epoch_snnl_loss=[]
        tmp_model.to(self.device)

        if stage == 2:
            for child in enumerate(model.children()):
                if child[0] <= 4:
                    for param in child[1].parameters():
                        param.requires_grad = False
                if child[0] == 4:
                    for subchild in enumerate(child[1].children()):
                        if subchild[0] == 2:
                            for subsubchild in enumerate(subchild[1].children()):
                                if subsubchild[0] == 3 or subsubchild[0] == 4:
                                    for param in subsubchild[1].parameters():
                                        param.requires_grad = True

        if stage == 3:
            for child in enumerate(model.children()):
                if child[0] <= 5:
                    for param in child[1].parameters():
                        param.requires_grad = False
                if child[0] == 5:
                    for subchild in enumerate(child[1].children()):
                        if subchild[0] == 3:
                            for subsubchild in enumerate(subchild[1].children()):
                                if subsubchild[0] == 3 or subsubchild[0] == 4:
                                    for param in subsubchild[1].parameters():
                                        param.requires_grad = True
        if stage == 4:
            for child in enumerate(model.children()):
                if child[0] <= 6:
                    for param in child[1].parameters():
                        param.requires_grad = False
                if child[0] == 6:
                    for subchild in enumerate(child[1].children()):
                        if subchild[0] == 5:
                            for subsubchild in enumerate(subchild[1].children()):
                                if subsubchild[0] == 3 or subsubchild[0] == 4:
                                    for param in subsubchild[1].parameters():
                                        param.requires_grad = True

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        gradients = []
        accumulated_gradients = None

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                legend = copy.deepcopy(labels)
                legend = torch.reshape(legend, (legend.shape[0], 1))
                tmp_label = torch.zeros(legend.shape[0], 10).to(self.device)
                tmp_label.scatter_(1, legend, 1)
                model.zero_grad()
                outputs, activa1,logit = model(images)
                images = torch.reshape(images, (images.shape[0], -1))

                hsic_y = hsic_normalized(logit, tmp_label, 5)
                hsic_x = hsic_normalized(activa1, images, 5)

                L2_regularization = 0
                ####### calculate L2 regularization term
                ###### '''''''

                if stage==1:
                    loss1 = self.criterion(outputs, labels)
                    loss2 = -self.args.lambda1_stage1*hsic_x-self.args.lambda2_stage1*hsic_y
                    loss3 = self.args.mu* L2_regularization
                    loss = loss1 +loss2  + loss3
                if stage ==2:
                    loss1 = self.criterion(outputs, labels)
                    loss2 = -self.args.lambda1_stage2 * hsic_x - self.args.lambda2_stage2 * hsic_y
                    loss3 = self.args.mu * L2_regularization
                    loss = loss1 + loss2 + loss3
                if stage ==3:
                    loss1 = self.criterion(outputs, labels)
                    loss2 = -self.args.lambda1_stage3 * hsic_x - self.args.lambda2_stage3 * hsic_y
                    loss3 = self.args.mu * L2_regularization
                    loss = loss1 + loss2 + loss3

                if stage ==4:
                    loss1 = self.criterion(outputs, labels)
                    loss2 = -self.args.lambda1_stage4 * hsic_x - self.args.lambda2_stage4 * hsic_y
                    loss3 = self.args.mu * L2_regularization
                    loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 50 == 0):
                    print('clients :{} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        idx,
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
        return model.state_dict(), 0, model, accumulated_gradients, model.state_dict(),self.sorted_idx


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        individual_loss = []

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs,activation = model(images)
            cri = nn.CrossEntropyLoss(reduction='none').to(self.device)
            batch_loss = cri(outputs, labels)
            # batch_loss = Self_cross_entropy(outputs,labels)
            # print('batch_loss =',batch_loss)
            # loss += batch_loss.item()
            individual_loss.extend(batch_loss.cpu().detach().numpy().tolist())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return individual_loss


def get_activations(model, inputs,activations):
    x = inputs
    for name,module in model.named_children():

        if name == 'fc':
            x = x.view(-1,512)
        x = module(x)

        if name in activations and activations[name] is None:
            activations[name] = x
        elif name in activations and activations[name] is not None:
            activations[name] = torch.cat([activations[name],x],dim=0)
    return activations

import numpy as np
class LabeledDataset(Dataset):
    def __init__(self, data, labels, label):
        indices = np.array(labels) == label
        data= data.transpose((0, 3, 1, 2))
        # indices = indices[:20]
        labels = np.array(labels)

        self.data = data[indices][:200]
        self.labels = labels[indices][:200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]

from torch.utils.data import ConcatDataset
from CKA import linear_CKA
# def test_inference_init(args, model, test_dataset,epoch,stage,test_loader):
#     """ Returns the test accuracy and loss.
#     """
#
#     test_data = test_dataset.data
#     test_labels = test_dataset.targets
#     label_to_loader = {}
#
#     total_sample = []
#     for label in range(10):
#         labeled_dataset = LabeledDataset(test_data, test_labels, label)
#         # label_to_loader[label] = DataLoader(labeled_dataset, batch_size=200, shuffle=False)
#         total_sample.append(labeled_dataset)
#     combined_dataset = ConcatDataset(total_sample)
#
#     sample_loader = DataLoader(combined_dataset, batch_size=2000, shuffle=False)
#
#     model.eval()
#     loss, total, correct = 0.0, 0.0, 0.0
#
#     device = 'cuda' if args.gpu else 'cpu'
#     criterion = nn.CrossEntropyLoss().to(device)
#     testloader = DataLoader(test_dataset, batch_size=1000,
#                             shuffle=False)
#
#     for batch_idx, (images, labels) in enumerate(testloader):
#     # for b in test_loader(0):
#     #     images, labels = b
#         # images, labels = images.to(device), labels.to(device)
#         images = images.float()
#
#         images, labels = images.to(device), labels.to(device)
#
#         # Inference
#         outputs,activation = model(images)
#         # batch_loss = criterion(outputs, labels)
#         # loss += batch_loss.item()
#
#         tsne_plot = None
#
#         if stage==1 and epoch%10==0:
#
#             if tsne_plot is not None:
#                 del tsne_plot
#
#             tsne_plot = Tsne_plot(f'motivation_stage1_{epoch}')
#             tsne_plot.plot(activation.detach().cpu().numpy(),labels=labels.detach().cpu().numpy())
#
#
#         if stage==2:
#             tsne_plot = Tsne_plot(f'curri_sample_stage2')
#             tsne_plot.plot(activation.detach().cpu().numpy(), labels=labels.detach().cpu().numpy())
#             if tsne_plot in locals():
#                 del tsne_plot
#
#         if stage==3:
#             tsne_plot = Tsne_plot(f'curri_sample_stage3')
#             tsne_plot.plot(activation.detach().cpu().numpy(),labels=labels.detach().cpu().numpy())
#             if tsne_plot in locals():
#                 del tsne_plot
#
#         if stage==4:
#             tsne_plot = Tsne_plot(f'curri_sample_stage4')
#             tsne_plot.plot(activation.detach().cpu().numpy(),labels=labels.detach().cpu().numpy())
#             if tsne_plot in locals():
#                 del tsne_plot
#         # Prediction
#         _, pred_labels = torch.max(outputs, 1)
#         pred_labels = pred_labels.view(-1)
#         correct += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)
#         break
#
#     accuracy = correct/total
#     return accuracy, 0,0,0
#     # return 0, 0
#
def test_inference_init(args, model, test_dataset,epoch,stage,test_loader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
    # for b in test_loader(0):
    #     images, labels = b
        images, labels = images.to(device), labels.to(device)
        # print('label shape=',labels.shape)

        legend = copy.deepcopy(labels)
        legend = torch.reshape(legend, (legend.shape[0], 1))
        # print('labels =',labels)
        tmp_label = torch.zeros(legend.shape[0], 10).to(device)
        tmp_label.scatter_(1, legend, 1)

    # Inference
        outputs,activa1,logit= model(images)
        # images = torch.reshape(images,(images.shape[0],-1))
        # pool = torch.nn.AdaptiveAvgPool2d((1,1))
        # images = pool(images)
        images = torch.reshape(images, (images.shape[0], -1))

        # if batch_idx==0:
        #     hsic_y1 = hsic_normalized(outputs,tmp_label,5)
        #     hsic_x1 = hsic_normalized(activa1,images, 5)
            # hsic_y2 = hsic_normalized(activa2, tmp_label, 5)
            # hsic_x2 = hsic_normalized(activa2, images, 5)
            # hsic_y3 = hsic_normalized(activa3, tmp_label, 5)
            # hsic_x3 = hsic_normalized(activa3, images, 5)
            # hsic_y4 = hsic_normalized(outputs, tmp_label, 5)
            # hsic_x4 = hsic_normalized(images, images, 5)
        # if batch_idx==0:
        #     snnL = SNNLCrossEntropy(temperature=100)
        #     snnl_loss = snnL.SNNL(images.cpu(), labels.cpu())



            # hsic_y = linear_CKA(activa.detach().cpu(), tmp_label.cpu())
            # hsic_x = linear_CKA(activa.detach().cpu(), images.cpu())


        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss



