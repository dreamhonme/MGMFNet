import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from data_loader.GetDataset_ISIC2018 import ISIC2018_dataset
from data_loader.GetDataset_ISIC2017 import ISIC2017_dataset
from data_loader.GetDataset_ISIC2016 import ISIC2016_dataset
from data_loader.GetDataset_PH import PH_dataset
from data_loader.ISIC2017 import ISIC2017_test_dataset
from data_loader.GetDataset_CHASE import MyDataset_CHASE
from model.DconnNet import DconnNet
import glob
import argparse
from torchvision import datasets, transforms
from solver import Solver
import torch.nn.functional as F
import torch.nn as nn
import cv2
from skimage.io import imread, imsave
import os
from data_loader.Polyp_Getdata import *
torch.cuda.set_device(1)  # GPU id
def parse_args():
    parser = argparse.ArgumentParser(
        description='DconnNet Training With Pytorch')

    # dataset info
    parser.add_argument('--dataset', type=str, default='isic2017_1',
                        help=' isic')

    parser.add_argument('--data_root', type=str, default='/home/ubuntu/Experiment/liyachao/Data/ISIC2018_npy_all_224_320',
                        help='dataset directory')
    parser.add_argument('--resize', type=int, default=[224, 320], nargs='+',
                        help='image size: [height, width]')

    # network option & hyper-parameters
    parser.add_argument('--num-class', type=int, default=1, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--lr-update', type=str, default='step',
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts')
    parser.add_argument('--lr-step', type=int, default=12,
                        help='define only when you select step lr optimization: what is the step size for reducing your lr')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)')

    parser.add_argument('--use_SDL', action='store_true', default=False,
                        help='set as True if use SDL loss; only for Retouch dataset in this code. If you use it with other dataset please define your own path of label distribution in solver.py')
    parser.add_argument('--folds', type=int, default=5,
                        help='define folds number K for K-fold validation')

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default=None,
                        help='put the path to resuming file if needed')
    # parser.add_argument('--weights', type=str, default='/home/ziyun/Desktop/project/BiconNet_codes/DconnNet/general/data_loader/retouch_weights/',
    #                     help='path of SDL weights')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--save-per-epochs', type=int, default=15,
                        help='per epochs to save')

    # evaluation only
    parser.add_argument('--test_only', action='store_true', default=True,
                        help='test only, please load the pretrained model')
    args = parser.parse_args()
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    return args


def main(args):
    # for exp_id in range(args.folds):
        exp_id = 3
        if args.dataset == 'isic':
            trainset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train',
                                        with_name=False)
            validset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='test',
                                        with_name=False)
        elif args.dataset == 'isic2017':
            trainset = ISIC2017_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train',
                                        with_name=False)
            validset = ISIC2017_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='test',
                                        with_name=False)
        elif args.dataset == 'isic2016':
            trainset = ISIC2016_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train',
                                        with_name=False)
            validset = ISIC2016_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='test',
                                        with_name=False)
        elif args.dataset == 'isic2017_1':
             validset = ISIC2017_test_dataset(dataset_root=args.data_root,
                                        with_name=False)
        elif args.dataset == 'PH':
            # trainset = PH_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train',
            #                             with_name=False)
            validset = PH_dataset(dataset_folder=args.data_root,
                                        with_name=False)
        # elif args.dataset == 'poylp':
        #     trainset = get_loader(image_root="/home/ubuntu/Experiment/liyachao/Data/CVC-ClinicDB/train/images", gt_root = "/home/ubuntu/Experiment/liyachao/Data/CVC-ClinicDB/train/groundtruth",batchsize=10,trainsize = 352)
        #     validset = test_dataset(image_root="/home/ubuntu/Experiment/liyachao/Data/CVC-ClinicDB/test/images", gt_root = "/home/ubuntu/Experiment/liyachao/Data/CVC-ClinicDB/test/groundtruth",testsize = 352)
        # else:
            ####  define how you get the data on your own dataset ######
            pass

        # train_loader = torch.utils.data.DataLoader(
        #     dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)  # 用于封装数据集并提供批量加载数据的迭代器
        val_loader = torch.utils.data.DataLoader(
            dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)

        # print("Train batch number: %i" % len(train_loader))
        print("Test batch number: %i" % len(val_loader))
        model = DconnNet(num_class=args.num_class).cuda()  # 实例化模型
        solver = Solver(args)
        best_model_path = os.path.join("/home/ubuntu/Experiment/liyachao/DconnNet/DconnNet-mainT", 'MGMS2017.pth')
        # if args.pretrained:
        if os.path.isfile(best_model_path):
            # model.load_state_dict(torch.load(args.pretrained,map_location = torch.device('cpu')))
            # model = model.cuda()
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda')))
            print(f"Loaded best model from {best_model_path}")
        else:
            raise ValueError("Best model file not found at {}".format(best_model_path))
        if args.test_only:
            solver.test_epoch(model, val_loader,1,3)


if __name__ == '__main__':
    args = parse_args()
    main(args)
