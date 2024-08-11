import os
import copy
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate,test_inference_init
from utils_cifar10 import get_dataset, average_weights, exp_details
import random
from Models_resnet import ResNet18_1,ResNet18_2,ResNet18_3,ResNet18_4

def distri_device():
    entire_net = []
    first_stage = []
    second_stage = []
    third_stage = []
    forth_stage = []

    ###according to the profile results to distribute memory resource
    ratio1 = len(entire_net) / 100
    ratio2 = len(first_stage) / 100
    ratio3 = len(second_stage) / 100
    ratio4 = len(third_stage) / 100
    ratio5 = len(forth_stage) / 100
    participant_ratio = [ratio1, ratio2, ratio3, ratio4, ratio5]

    return participant_ratio, first_stage, second_stage, third_stage, forth_stage

if __name__ == '__main__':
    start_time = time.time()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    txtpath = 'NeuLite_ResNet34_CIFAR10_IID.txt'
    f_acc = open(txtpath, 'a')
    f_acc.write(f'agrs = {args} \n')
    f_acc.flush()

    teacher_model = ResNet18_4()
    participant_ratio, first_stage, second_stage, third_stage, forth_stage = distri_device()
    f_acc.write(f'participant_ratio : {participant_ratio}  \n')
    f_acc.write(f'first stage user :{first_stage}\n')
    f_acc.write(f'second stage user :{second_stage}\n')
    f_acc.write(f'third stage user :{third_stage}\n')
    f_acc.write(f'forth stage user :{forth_stage}\n')
    f_acc.flush()

    train_dataset, test_dataset, user_groups = get_dataset(args)

    test_loader = None
    device = 'cuda'
    global_acc = []
    total_hsic_X = []
    total_hsic_Y = []
    best_acc = 0

    tmp_model = ResNet18_4().to(device)

    for epoch in range(1,1600):
        if epoch % 4 == 1:
            idx_users = np.random.choice(first_stage,10,replace=False)
            global_model = ResNet18_1().to(device)
            stage = 1
            if epoch > 1:
                global_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage1_2.pth'))
            if epoch > 4:
                tmp_model = ResNet18_4()
                tmp_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage4_2.pth'))
                for child in enumerate(tmp_model.children()):
                    if child[0] == 4:
                        weight4 = child[1].state_dict()
                for child in enumerate(global_model.children()):
                    if child[0] == 4:
                        child[1].load_state_dict(weight4)
        if epoch % 4 == 2:
            idx_users = np.random.choice(second_stage,10,replace=False)
            global_model = ResNet18_2().to(device)
            stage = 2
            if epoch > 2:
                global_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage2_2.pth'))
            if epoch >= 2:
                tmp_model = ResNet18_1()
                tmp_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage1_2.pth'))
                for child in enumerate(tmp_model.children()):
                    if child[0] == 0:
                        weight0 = child[1].state_dict()
                    if child[0] == 1:
                        weight1 = child[1].state_dict()
                    if child[0] == 2:
                        weight2 = child[1].state_dict()
                    if child[0] == 3:
                        weight3 = child[1].state_dict()
                    if child[0] == 4:
                        weight4 = child[1].state_dict()
                for child in enumerate(global_model.children()):
                    if child[0] == 0:
                        child[1].load_state_dict(weight0)
                    if child[0] == 1:
                        child[1].load_state_dict(weight1)
                    if child[0] == 2:
                        child[1].load_state_dict(weight2)
                    if child[0] == 3:
                        child[1].load_state_dict(weight3)
                    if child[0] == 4:
                        child[1].load_state_dict(weight4)
            if epoch > 4:
                tmp_model = ResNet18_4()
                tmp_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage4_2.pth'))
                for child in enumerate(tmp_model.children()):
                    if child[0] == 5:
                        weight5 = child[1].state_dict()

                for child in enumerate(global_model.children()):
                    if child[0] == 5:
                        child[1].load_state_dict(weight5)
        if epoch % 4 == 3:
            idx_users = np.random.choice(third_stage,10,replace=False)

            global_model = ResNet18_3().to(device)
            stage = 3
            if epoch > 3:
                global_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage3_2.pth'))

            if epoch >= 3:
                tmp_model = ResNet18_2()
                tmp_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage2_2.pth'))
                for child in enumerate(tmp_model.children()):
                    if child[0] == 0:
                        weight0 = child[1].state_dict()
                    if child[0] == 1:
                        weight1 = child[1].state_dict()
                    if child[0] == 2:
                        weight2 = child[1].state_dict()
                    if child[0] == 3:
                        weight3 = child[1].state_dict()
                    if child[0] == 4:
                        weight4 = child[1].state_dict()
                    if child[0] == 5:
                        weight5 = child[1].state_dict()

                for child in enumerate(global_model.children()):
                    if child[0] == 0:
                        child[1].load_state_dict(weight0)
                    if child[0] == 1:
                        child[1].load_state_dict(weight1)
                    if child[0] == 2:
                        child[1].load_state_dict(weight2)
                    if child[0] == 3:
                        child[1].load_state_dict(weight3)
                    if child[0] == 4:
                        child[1].load_state_dict(weight4)
                    if child[0] == 5:
                        child[1].load_state_dict(weight5)
            if epoch > 4:
                tmp_model = ResNet18_4()
                tmp_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage4_2.pth'))
                for child in enumerate(tmp_model.children()):
                    if child[0] == 6:
                        weight6 = child[1].state_dict()

                for child in enumerate(global_model.children()):
                    if child[0] == 6:
                        child[1].load_state_dict(weight6)
        if epoch % 4 == 0:
            idx_users = np.random.choice(forth_stage,10,replace=False)

            global_model = ResNet18_4().to(device)
            stage = 4
            if epoch > 4:
                global_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage4_2.pth'))

            if epoch >= 4:
                tmp_model = ResNet18_3()
                tmp_model.load_state_dict(torch.load('NeuLite_cifar10_iid_stage3_2.pth'))
                for child in enumerate(tmp_model.children()):
                    if child[0] == 0:
                        weight0 = child[1].state_dict()
                    if child[0] == 1:
                        weight1 = child[1].state_dict()
                    if child[0] == 2:
                        weight2 = child[1].state_dict()
                    if child[0] == 3:
                        weight3 = child[1].state_dict()
                    if child[0] == 4:
                        weight4 = child[1].state_dict()
                    if child[0] == 5:
                        weight5 = child[1].state_dict()
                    if child[0] == 6:
                        weight6 = child[1].state_dict()

                for child in enumerate(global_model.children()):
                    if child[0] == 0:
                        child[1].load_state_dict(weight0)
                    if child[0] == 1:
                        child[1].load_state_dict(weight1)
                    if child[0] == 2:
                        child[1].load_state_dict(weight2)
                    if child[0] == 3:
                        child[1].load_state_dict(weight3)
                    if child[0] == 4:
                        child[1].load_state_dict(weight4)
                    if child[0] == 5:
                        child[1].load_state_dict(weight5)
                    if child[0] == 6:
                        child[1].load_state_dict(weight6)
        f_acc.write('-----------------------------------------------------------------\n')
        f_acc.write(f'global round {epoch} :{idx_users}  \n')
        f_acc.flush()
        local_weights, local_losses = [], []
        after_scalar = {}
        for idx in idx_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger,teacher_model=teacher_model,stage=stage,epoch=epoch)
            w, loss,update_local_model,accumulated_gradients,model_param,sorted_idx = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch,idx = idx,stage=stage,tmp_model=tmp_model)
            local_weights.append(copy.deepcopy(w))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        test_acc0, test_loss0 = test_inference_init(args, global_model,test_dataset,epoch,stage,test_loader)

        if stage == 1:
            torch.save(global_model.state_dict(), f'NeuLite_cifar10_iid_stage1_2.pth')
        if stage == 2:
            torch.save(global_model.state_dict(), f'NeuLite_cifar10_iid_stage2_2.pth')
        if stage == 3:
            torch.save(global_model.state_dict(), f'NeuLite_cifar10_iid_stage3_2.pth')
        if stage == 4:
            torch.save(global_model.state_dict(), f'NeuLite_cifar10_iid_stage4_2.pth')
        global_acc.append(test_acc0)
        print('global_acc =',global_acc)
        f_acc.write(f'global_acc = {global_acc} \n')
        f_acc.write(f'HSIC_X = {total_hsic_X} \n')
        f_acc.write(f'HSIC_Y = {total_hsic_Y} \n')
        f_acc.write('***************************\n')
        f_acc.flush()