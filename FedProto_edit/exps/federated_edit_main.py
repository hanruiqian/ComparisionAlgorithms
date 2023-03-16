#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import pandas as pd
import csv

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_inference_new_het_lt_edit
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem, agg_func_edit, proto_aggregation_edit

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    #idxs_users用户索引
    idxs_users = np.arange(args.num_users)
    res_train_acc_list, res_train_loss1_list, res_train_loss2_list, res_train_total_loss_list = [], [], [], []
    res_test_acc_list, res_test_loss1_list, res_test_loss2_list, res_test_total_loss_list = [], [], [], []
    train_loss, train_accuracy = [], []
    test_loss, test_acc = [], []
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        global_train_acc_list, global_train_loss1_list, global_train_loss2_list, global_train_total_loss_list = [], [], [], []
        label_num = dict()
        print(f'\n | Global Training Round : {round + 1} |\n')
        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            # 根据平均值得到本地原型
            agg_protos, local_label_num = agg_func_edit(protos)
            label_num[idx] = local_label_num
            # 对于客户端i的该轮本地训练的weight
            local_weights.append(copy.deepcopy(w))
            # 对于客户端i的该轮本地训练loss
            local_losses.append(copy.deepcopy(loss['total']))
            # 对于客户端i的该轮本地训练的proto， local_protos{idx : { class_type : local_proto }}
            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            # 对于客户端i的该轮本地训练最后一个batch的acc
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']
            global_train_acc_list.append(acc)
            global_train_loss1_list.append(loss['1'])
            global_train_loss2_list.append(loss['2'])
            global_train_total_loss_list.append(loss['total'])

        global_train_acc = np.mean(global_train_acc_list)
        global_train_loss1 = np.mean(global_train_loss1_list)
        global_train_loss2 = np.mean(global_train_loss2_list)
        global_train_total_loss = np.mean(global_train_total_loss_list)
        print('For all users, mean of train acc is {:.5f}'.format(global_train_acc))
        print('For all users, mean of train loss1 is {:.5f}'.format(global_train_loss1))
        print('For all users, mean of train loss2 is {:.5f}'.format(global_train_loss2))
        print('For all users, mean of train loss is {:.5f}'.format(global_train_total_loss))
        res_train_acc_list.append(global_train_acc)
        res_train_loss1_list.append(global_train_loss1)
        res_train_loss2_list.append(global_train_loss2)
        res_train_total_loss_list.append(global_train_total_loss)

        # update global weights
        local_weights_list = local_weights
        # 对模型本地权重进行更新
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
        
        t_local_model_list = local_model_list
        for idx in idxs_users:
            local_model = copy.deepcopy(t_local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            t_local_model_list[idx] = local_model

        # update global weights
        # 得到全局类原型global_protos{ class_type : proto }
        if args.explore_lgs is True:
            global_protos = proto_aggregation_edit(local_protos, label_num)
        else:
            global_protos = proto_aggregation(local_protos)
        
        #该轮的全局loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # ---------------------
        acc_list, loss1_list, loss2_list, total_loss_list =  test_inference_new_het_lt_edit(args, t_local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
        global_test_acc = np.mean(acc_list)
        global_test_loss1 = np.mean(loss1_list)
        global_test_loss2 = np.mean(loss2_list)
        global_total_loss = np.mean(total_loss_list)
        print('For all users, mean of test acc is {:.5f}'.format(global_test_acc))
        print('For all users, mean of test loss1 is {:.5f}'.format(global_test_loss1))
        print('For all users, mean of test loss2 is {:.5f}'.format(global_test_loss2))
        print('For all users, mean of total loss is {:.5f}'.format(global_total_loss))
        res_test_acc_list.append(global_test_acc)
        res_test_loss1_list.append(global_test_loss1)
        res_test_loss2_list.append(global_test_loss2)
        res_test_total_loss_list.append(global_total_loss)
        # ---------------------
    # acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    # print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    # print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    # print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
    
    # 保存CSV文件
    train_frame = pd.DataFrame({'Train Acc' : res_train_acc_list, 'Train Loss1' : res_train_loss1_list, 'Train Loss2' : res_train_loss2_list, 'Train Loss' : res_train_total_loss_list})
    test_frame = pd.DataFrame({'Test Acc' : res_test_acc_list, 'Test Loss1' : res_test_loss1_list, 'Test Loss2' : res_test_loss2_list, 'Test Loss' : res_test_total_loss_list})
    file1_n = './Train Res ' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.rounds) + 'r' + str(args.num_users) + 'u' + '.csv'
    file2_n = './Test Res ' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.rounds) + 'r' + str(args.num_users) + 'u' + '.csv'
    train_frame.to_csv(file1_n, sep=',')
    test_frame.to_csv(file2_n, sep=',')
    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        # 每个epoch的平均准确率
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    #为每个客户端随机生成n-way
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        #为每个客户端随机生成k-shot
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)

    #user_groups 每个客户端根据n-way k-shot划分的结果（训练集索引）
    #user_groups_lt 每个客户端根据n-way k-shot划分的结果（测试集索引）
    #classes_list 每个客户端包含的类别
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            #异构模型
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            #非异构模型
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)