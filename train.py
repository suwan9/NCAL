from __future__ import print_function
from cProfile import label
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import open_entropy,entropync,mixup,ova_loss2
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models,imageSavePIL
from eval import test, evaluate_sourcecentroid, evaluate_ttcentroid
import argparse
import loss
#import os
#import sys
#import argparse
from os.path import join
import models
import torch.optim as optim

###############输出图像
# open-cv library is installed as cv2 in python
# import cv2 library into this program
import cv2
# import numpy library as np
import numpy as np
from PIL import Image
import math
import copy

from models.LinearAverage import LinearAverage
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml',
                    help='/path/to/config/file')

parser.add_argument('--source_data', type=str,
                    default='./utils/source_list.txt',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./utils/target_list.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='office',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=False, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/ova_model",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.1,
                    help='weight factor for adaptation')
#数据增强中增加
parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
#parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')##递减策略 
parser.add_argument('--pretrain_model_path', type=str, default='/home1/suwan/Project/OVANet1/tsne', help="")
##/home1/suwan/Project/OVANet1/pretrainedm
##GAN
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')

args = parser.parse_args()

config_file = args.config
conf = yaml.load(open(config_file),Loader=yaml.FullLoader)
save_config = yaml.load(open(config_file),Loader=yaml.FullLoader)
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open1 = n_total - n_share - n_source_private > 0
#open1 = True #############c本来为False
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

source_loader, target_loader, test_loader, target_folder, \
dataset_labeled, labeled_trainloader, s_loader = get_dataloaders(inputs)
#

#dataset_unlabeled, unlabeled_trainloader,
logname = log_set(inputs)

G, C1, C2, CA, opt_g, opt_c, opt_c1, \
param_lr_g, param_lr_c, param_lr_c1, ad_net, opt_a, param_lr_ad,decreasing_lr,random_layer = get_models(inputs,args)
ndata = target_folder.__len__()
#
## Memory
lemniscate = LinearAverage(2048, ndata, conf.model.temp, conf.train.momentum).cuda()



def train():
    cor_fc = []
    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.BCEWithLogitsLoss().cuda()#nn.BCELoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader) 
    
    h_score_best = 0
    acc_all_best = 0
    common_acc1 = 0
    common_acc2 = 0
    open_acc1 = 0
    open_acc2 = 0
    clean_num = np.zeros(num_class)#officehome-15     #office31-20
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()

        ad_net.train()
        CA.train() 
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)


        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        

        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)

        inv_lr_scheduler(param_lr_c1, opt_c1, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_ad, opt_a, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)

        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        label_t1 = data_t[1]
        index_t = data_t[2]
       
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        index_t = Variable(index_t.cuda())
        label_st = Variable(label_st.cuda())



        opt_g.zero_grad()
        opt_c.zero_grad()
        opt_c1.zero_grad()
        opt_a.zero_grad()

        C2.module.weight_norm()#########
        
        feat = G(img_s)
        out_s = C1(feat)
        out_open = C2(feat)


        out_open1 = out_open.view(img_s.size(0), 2, -1)
        logits_o1 = F.softmax(out_open1, 1)
        logits_o11 = F.softmax(out_open1, 2)
        
        out_openb = out_open.view(out_s.size(0), 2, -1)
        out_openb = F.softmax(out_openb, 2)
        loss_s1 = criterion(out_openb[:, 1, :], label_s)
        out_open = out_open.view(out_s.size(0), 2, -1)

        
        open_loss_pos, open_loss_neg, open_loss_negm = ova_loss2(out_open, feat, label_s, C2)
        
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)  

        
        all = loss_s1 + loss_open #    ###########allin   
           
        log_string = 'Train {}/{} \t ' \
                     'Loss Source: {:.4f} ' \
                     'Loss Open: {:.4f} ' \
                     'Loss Open Source Positive: {:.4f} ' \
                     'Loss Open Source Negative: {:.4f} '
        log_values = [step, conf.train.min_step,
                      loss_s.item(),  loss_open.item(),
                      open_loss_pos.item(), open_loss_neg.item()]
        if not args.no_adapt:
            feat_t = G(img_t)
            out_open_t = C2(feat_t)
            out_open_t = out_open_t.view(img_t.size(0), 2, -1)
            logits_ou1 = F.softmax(out_open_t, 2)
            
            logits_ou2 = F.softmax(out_open_t, 1)
            logits_ou22 = F.softmax(out_open_t, 1)
            #0轴为样本，1轴二分类器之间的softmax，
            open_label = logits_ou2[:, 1, :].data.max(1)[1]#open_label
            
            ent_open = open_entropy(out_open_t)
            all += args.multi * ent_open  #
            log_values.append(ent_open.item())
            log_string += "Loss Open Target: {:.6f}"

       
        if step > 2000 and step % 10 == 0: #50
            open_loss_pos,open_loss_neg2 = mixup(all_unfeat, label_s, label_st, C2,clean_feat,clean_la,cor_fc)
            
            loss_open2 = open_loss_pos + open_loss_neg2     
    
        featurest = torch.cat((feat, feat_t), dim=0) 
        aa = logits_o1[:, 1, :]#二分类    
        bb = logits_ou22[:, 1, :]#二分类       
        outputst = torch.cat((aa, bb), dim=0)

        transfer_loss = loss.CDAN([featurest, outputst], ad_net,label_s,open_label,step,cor_fc, None, None, random_layer)
        all += 0.5*transfer_loss    
       
        feat_t1 = F.normalize(feat_t)
        feat_mat = lemniscate(feat_t1, index_t)
        feat_mat[:, index_t] = -1 / conf.model.temp
        feat_mat2 = torch.matmul(feat_t1,
                                 feat_t1.t()) / conf.model.temp
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().cuda()
       
        feat_mat2.masked_fill_(mask, -1 / conf.model.temp)
       
        if step > 1000 :
            initck1 = torch.from_numpy(np.float16(initck)).cuda()        
            feat_initck = torch.matmul(feat_t1,
                                    initck1.t()) / conf.model.temp
          
            feat_initckall = []
            for i in range(feat_initck.size(0)):
                feat_initckall.append([feat_initck[i].tolist()])
            feat_initckall = torch.tensor(feat_initckall).cuda()
           
       
        if step <= 1000 :
            loss_nc = conf.train.eta * entropync(torch.cat([logits_ou2[:, 1, :], feat_mat,  #logits_ou1
                                                            feat_mat2], 1))
        else:
            loss_nc = conf.train.eta * entropync(torch.cat([logits_ou2[:, 1, :], feat_mat, feat_initckall, #logits_ou1
                                                            feat_mat2], 1))
                                           
        all += 0.5*loss_nc 

        with amp.scale_loss(all, [opt_g, opt_c, opt_c1, opt_a]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()

        opt_c1.step()
        opt_a.step()
        opt_a.zero_grad()
        opt_c1.zero_grad()

        opt_g.zero_grad()
        opt_c.zero_grad()
        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))

       
        if step >= 0 and step % 50 == 0:
           
            initc_s,labelset_s, in_ss, out_ss, avg_s1, cc = evaluate_sourcecentroid(s_loader, G, args, C2)         
            G.train()          
            C2.train()
          
        if step >= 0 and step % 50 == 0:
           
            initc,labelset,all_unfeat,clean_feat,clean_la,clean_num,cor_fc,uninitc,initck = evaluate_ttcentroid(s_loader, test_loader, G, C2, args,n_share,num_class,clean_num,initc_s,labelset_s,cor_fc,step) #, source_classes, args
            
            G.train()
            C2.train()
       

        if step > 0 and step % conf.test.test_interval == 0:
            acc_o, h_score, h_score_best,acc_all_best,common_acc1,common_acc2,open_acc1,open_acc2 = test(initc_s, num_class,step, test_loader, logname, n_share, G,
                                  [C1, C2], h_score_best, acc_all_best,initc,common_acc1,common_acc2,open_acc1,open_acc2,open=open1)

            print("acc all %s h_score %s " % (acc_o, h_score))
            G.train()
            C1.train()
            C2.train()
            if args.save_model:
                save_path = "%s_%s.pth"%(args.save_path, step)
                save_model(G, C1, C2, save_path)
            

train()
