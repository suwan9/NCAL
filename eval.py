from cProfile import label
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score

import copy
from os.path import join
import argparse


from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from common.utils.analysis import tsne
import os.path as osp




def feat_get(step, G, Cs, dataset_source, dataset_target, save_path,
             ova=True):
    G.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s = G(img_s)


            if batch_idx == 0:
                feat_all_s = feat_s.data.cpu().numpy()
                label_all_s = label_s.data.cpu().numpy()
            else:
                feat_s = feat_s.data.cpu().numpy()
                label_s = label_s.data.cpu().numpy()
                feat_all_s = np.r_[feat_all_s, feat_s]
                label_all_s = np.r_[label_all_s, label_s]
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat_t = G(img_t)

            out_t = Cs[0](feat_t)
            pred = out_t.data.max(1)[1]
            out_t = F.softmax(out_t)
            if ova:
                out_open = Cs[1](feat_t)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1), 1)
                tmp_range = torch.range(0, out_t.size(0) - 1).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                weights_open = Cs[1].module.fc.weight.data.cpu().numpy()
            else:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)

            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
                unk_all = pred_unk.data.cpu().numpy()
                pred_all = pred.data.cpu().numpy()
                pred_all_soft = out_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                pred_unk = pred_unk.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
                unk_all = np.r_[unk_all, pred_unk]
                pred_all = np.r_[pred_all, pred.data.cpu().numpy()]
                pred_all_soft = np.r_[pred_all_soft, out_t.data.cpu().numpy()]

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "save_%s_ova_%s_target_feat.npy" % (step, ova)), feat_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_anom.npy" % (step, ova)), unk_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_pred.npy" % (step, ova)), pred_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_soft.npy" % (step, ova)), pred_all_soft)
    np.save(os.path.join(save_path, "save_%s_ova_%s_source_feat.npy" % (step, ova)), feat_all_s)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_label.npy" % (step, ova)), label_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_source_label.npy" % (step, ova)), label_all_s)
    if ova:
        np.save(os.path.join(save_path, "save_%s_ova_%s_weight.npy" % (step, ova)), weights_open)

###################### CMU #################
class AccuracyCounter:
    
    def __init__(self, length):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def h_score(self):
        self.clear_zero()
        common_acc = np.mean(self.Ncorrect[0:-1] / self.Ntotal[0:-1])
        open_acc = self.Ncorrect[-1] / self.Ntotal[-1]
        return 2 * common_acc * open_acc / (common_acc + open_acc)
################################################


def test(initc_s, num_class, step, dataset_test, name, n_share, G, Cs, h_score_best, acc_all_best, initc,common_acc1,common_acc2,open_acc1,open_acc2 ,
         open_class = None, open=False, entropy=False, thr=None):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))

    per_class_fc_num = np.zeros((n_share + 1))
    per_class_c = np.zeros((num_class+1))    #office31-32   officehome-66   13
    per_class_logit = np.zeros((num_class+1))   #office31-32   officehome-66
    per_class_inlabel = [[] for _ in range((num_class+1))]  #office31-32   officehome-66

    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    cclass_list = [i for i in range(n_share)]
    label_al = []
    label_c = []
    label_ova = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)

            
            label_al.append(label_t)
           
           

            out_t = F.softmax(out_t)
            
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                
                out_logit = out_open.view(out_t.size(0), 2, -1)
               
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
               
                
                tmp_range = torch.range(0, out_t.size(0)-1).long().cuda()#生成【0，1,2，....，35】
                
                open_label = out_open[:, 1, :].data.max(1)[1]
                
                open_p = out_open[:, 1, :].data.max(1)[0]#已知概率最大值的值=自信度
                open_logit = out_logit[:, 1, :].data.max(1)[0]#已知logit最大值的值
                open_logit_un = out_logit[:, 0, :].data.max(1)[0]#未知logit最大值的值
                open_unlabel = out_logit[:, 0, :].data.max(1)[1]#未知logit最大值的下标
                
                open_logit_sum = [0 for i in range(out_logit.size(0))]
                for x in range(out_logit.size(0)):
                    open_logit_sum[x] = torch.sum(out_logit[x][0])#未知logit求和    
                #print(open_logit.size(),'open_logit.size()')
                open_logit_sum = torch.tensor(open_logit_sum)
                #print(open_logit_sum,open_logit_sum.size(),'open_logit_sum.size()')
               
                pred = open_label ######### 开集 
                pred_un = out_open[:, 1, :].data.max(1)[1] ######### 开集 
                correct_close += pred.eq(label_t.data).cpu().sum()
                #'''
                label_ova.append(open_label)
                #print(label_ova,'label_ova')

                pred_unk = out_open[tmp_range, 0, pred]
                #print(pred_unk,'pred_unk')
                ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]#未知类样本下标   0.5
                #print(ind_unk,'ind_unk')
            pred[ind_unk] = open_class
            pred_un[ind_unk] = open_unlabel[ind_unk]
            #print(pred,'pred1')
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            pred_un = pred_un.cpu().numpy()
            k = label_t.data.size()[0]
            #input()
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)

                ######### C + logit ####################
                f_ind = np.where(pred == t)
                incorrect_ind = np.where(pred[t_ind[0]] != t)
                #per_class_inlabel[i].append(pred[incorrect_ind[0]]) #本该在的类别里##youwenti
                #########################################

                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
                per_class_fc_num[i] += float(len(f_ind[0]))   #分入每类的样本总计
            size += k
            if 0==0:#open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()#pred#
                    pred_ent = entr

                    pred_allopen = pred
                    pred_unall = pred_un
                    feat_all = feat.cpu().numpy()
                    open_p_all = open_p.cpu().numpy()
                    open_logit_all = open_logit.cpu().numpy()
                    open_logit_unall = open_logit_un.cpu().numpy()
                    logit_sum_all = open_logit_sum.cpu().numpy()#未知logit求和 
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]#np.r_[pred_all, pred]#
                    label_all = np.r_[label_all, label_t]

                    pred_allopen = np.r_[pred_allopen, pred]
                    pred_unall = np.r_[pred_unall, pred_un]
                    feat_all = np.r_[feat_all, feat.cpu().numpy()]
                    open_p_all = np.r_[open_p_all, open_p.cpu().numpy()]
                    open_logit_all = np.r_[open_logit_all, open_logit.cpu().numpy()]
                    open_logit_unall = np.r_[open_logit_unall, open_logit_un.cpu().numpy()]
                    logit_sum_all = np.r_[logit_sum_all, open_logit_sum.cpu().numpy()]
                    
    counters2 = AccuracyCounter(len(class_list))
        
    for (each_indice, each_label) in zip(pred_allopen, label_all):
        if each_label in cclass_list:
            counters2.add_total(each_label)
            if each_indice == each_label:
                counters2.add_correct(each_label)
        else:
            counters2.add_total(-1)
            if each_indice == each_label:
                counters2.add_correct(-1)
    #best_acc3 = max(counters2.mean_accuracy(), best_acc3)
    best_acc3 = counters2.mean_accuracy()
    best_h_score3 = counters2.h_score()
    print('---CMU ACC---')
    print(counters2.each_accuracy())
    print(counters2.mean_accuracy(),'accuracy()')
    print(counters2.h_score(),'h_score()')
    #print(best_acc3,'best_acc3')
    common_acc = np.mean(counters2.Ncorrect[0:-1] / counters2.Ntotal[0:-1])
    open_acc = counters2.Ncorrect[-1] / counters2.Ntotal[-1]
   
    per_class_in = []
    per_unclass_logit = np.zeros(2)
    per_class_num1 = np.zeros(num_class+1) #office31-32   officehome-66   vi-13
    per_class_num2 = np.zeros(2)

    logit_mean = np.mean(open_logit_all)##已知logit的整体平均
    unlogit_mean = np.mean(open_logit_unall)##已知logit的整体平均
    class_numl_all = np.zeros(num_class+1)  #office31-32   officehome-66
    class_numl_cor = np.zeros(num_class+1)

    #counters2 = AccuracyCounter(len(class_list))        
    for (each_indice, each_label, cc,maxl,suml,maxunl) in zip(pred_allopen, label_all, open_p_all,open_logit_all,logit_sum_all,open_logit_unall):
        per_class_c[each_indice] += cc
        per_class_logit[each_indice] += maxl
        
        per_class_num1[each_indice] += 1
        if each_indice != each_label:
            per_class_in.append([cc,maxl,suml,each_indice, each_label])  #本该在的类别里
        if each_label in cclass_list:    
            per_unclass_logit[0] += suml
            per_class_num2[0] += 1
            if maxl < logit_mean:
                class_numl_all[each_label] += 1
                if each_indice == each_label:
                    class_numl_cor[each_label] += 1

        else:
            per_unclass_logit[1] += suml
            per_class_num2[1] += 1
            if maxunl < unlogit_mean:
                class_numl_all[-1] += 1
                if each_indice == each_label:
                    class_numl_cor[-1] += 1
            #取最后一个元素
        

    per_class_c1 = per_class_c / per_class_num1 
    per_class_logit1 = per_class_logit / per_class_num1
    per_unclass_logit1 = per_unclass_logit / per_class_num2
   
    if best_acc3 > acc_all_best:
        acc_all_best = best_acc3
        common_acc1 = common_acc
        open_acc1 = open_acc
    if best_h_score3 > h_score_best:
        h_score_best = best_h_score3
        common_acc2 = common_acc
        open_acc2 = open_acc
    
        best_classifier = copy.deepcopy(G.state_dict()) 
        best_esem = copy.deepcopy(Cs[0].state_dict()) 
        #with open('record/classifier_best_ct.pkl', 'wb') as f:
        torch.save(best_classifier, 'record/classifier_best_ct.pkl')
        #with open(join('./record', 'esem_best_ct.pkl'), 'wb') as f:
            #torch.save(best_esem, f)
        print('save ok!')
    #best_acc3 best_h_score3
    
    return best_acc3, best_h_score3, h_score_best, acc_all_best,common_acc1,common_acc2,open_acc1,open_acc2


def select_threshold(pred_all, conf_thr, label_all,
                     class_list, thr=None):
    num_class  = class_list[-1]
    best_th = 0.0
    best_f = 0
    best_known = 0
    best_unknown = 0
    if thr is not None:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > thr)[0]
        pred_class[ind_unk] = num_class
        return accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class)
    ran = np.linspace(0.0, 1.0, num=20)
    conf_thr = conf_thr / conf_thr.max()
    scores = []
    for th in ran:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > th)[0]
        pred_class[ind_unk] = num_class
        score, known, unknown = h_score_compute(label_all, pred_class,
                                                class_list)
        scores.append(score)
        if score > best_f:
            best_th = th
            best_f = score
            best_known = known
            best_unknown = unknown
    mean_score = np.array(scores).mean()
    print("best known %s best unknown %s "
          "best h-score %s"%(best_known, best_unknown, best_f))
    return best_th, best_f, mean_score


def h_score_compute(label_all, pred_class, class_list):
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros((len(class_list))).astype(np.float32)
    for i, t in enumerate(class_list):
        t_ind = np.where(label_all == t)
        correct_ind = np.where(pred_class[t_ind[0]] == t)
        per_class_correct[i] += float(len(correct_ind[0]))
        per_class_num[i] += float(len(t_ind[0]))
    open_class = len(class_list)
    per_class_acc = per_class_correct / per_class_num
    known_acc = per_class_acc[:open_class - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    return h_score, known_acc, unknown





  
def evaluate_ttcentroid(s_loader: DataLoader, val_loader: DataLoader, model, esem,args: argparse.Namespace,n_share,num_class,clean_num,initc_s,labelset_s,cor_fc,step):
    model.eval()
    esem.eval()
    #all_output = list()
    #cnt = 0
    class_list = [i for i in range(num_class)]
    start_test = True
    start_tests = True
    per_class_fc_num = np.zeros((num_class))
    avg = np.zeros((num_class))
    with torch.no_grad():
        ########### 源域 #############
        for i, data in enumerate(s_loader): #(images, labels)
            img_s, labels = data[0].cuda(), data[1].cuda()
            f = model(img_s) 
            y_1 = esem(f)

            out_open1 = y_1.view(img_s.size(0), 2, -1)
            logits_o1 = F.softmax(out_open1, 2)
            #这里的dim=0其实就是张量的0轴，dim=1就是张量的1轴，按照二分类求softmax。2轴为20公共类之间的softmax
            all_soft = logits_o1[:, 1, :]
            #[:, 1, :]的公共类
            #全部的特征
            if(start_tests):
                all_fs = f.float().cpu()
                all_sms= y_1.float().cpu()
                all_labels = labels.float()
                #all_softmaxs = all_soft.float().cpu()
                start_tests = False
            else:
                all_fs = torch.cat((all_fs, f.float().cpu()), 0)#按行拼接
                all_sms = torch.cat((all_sms, y_1.float().cpu()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)
                #all_softmaxs = torch.cat((all_softmaxs, all_soft.float().cpu()), 0)


        ########### 目标域 #############
        for i, data in enumerate(val_loader): #(images, labels)
            img_s, labels = data[0].cuda(), data[1].cuda()
            #images = images.to(device)
            f = model(img_s)
            
            y_1 = esem(f)

            open_class = int(y_1.size(1)/2)

            out_open1 = y_1.view(img_s.size(0), 2, -1)
            logits_o1 = F.softmax(out_open1, 2)
            
            all_soft = logits_o1[:, 1, :]

            out_open = F.softmax(y_1.view(img_s.size(0), 2, -1),1)
            
            open_label = logits_o1[:, 1, :].data.max(1)[1]
            open_label1 = logits_o1[:, 1, :].data.max(1)[1]#已知类
            

            open_label2 = out_open[:, 0, :].data.min(1)[0]
           

            pred = open_label
            
            tmp_range = torch.range(0, img_s.size(0)-1).long().cuda()
           
            pred_unk = out_open[tmp_range, 0, pred]
            #print(pred_unk,'pred_unk')
            ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]
            pred[ind_unk] = open_class

            
            for j, t in enumerate(class_list):
                
                f_ind = np.where(pred.cpu() == t)
               
                
                per_class_fc_num[j] += float(len(f_ind[0]))   
      
            if(start_test):
                all_f = f.float().cpu()
                all_sm= y_1.float().cpu()
                all_label = labels.float()
                all_pred = pred.cpu()#.numpy()
                all_pred1 = open_label1.cpu()#.numpy() open_label2
                all_pred2 = open_label2.cpu()
                all_softmax = all_soft.float().cpu()
                out_open_all = out_open1.float().cpu()
                start_test = False
            else:
                all_f = torch.cat((all_f, f.float().cpu()), 0)#按行拼接
                all_sm = torch.cat((all_sm, y_1.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_softmax = torch.cat((all_softmax, all_soft.float().cpu()), 0)
                all_pred = torch.cat((all_pred, pred.cpu()), 0)#.numpy()
                all_pred1 = torch.cat((all_pred1, open_label1.cpu()), 0)#.numpy()
                all_pred2 = torch.cat((all_pred2, open_label2.cpu()), 0)#.numpy()
                out_open_all = torch.cat((out_open_all, out_open1.cpu()), 0)#.numpy()
    
    all_fy = all_f.tolist()

    all_f = (all_f.t() / torch.norm(all_f, p=2, dim=1)).t() #重要
    all_fs = (all_fs.t() / torch.norm(all_fs, p=2, dim=1)).t() #重要

    unfeat = []
    for i in range(len(all_pred)):
        if all_pred [i] == open_class:
            if all_pred2[i] >= 0.97:   ##0.97
                unfeat.append(all_fy[i])
 
    qq = open_class#+1
    #print(qq,'qq')
    pknow = np.zeros((qq))
    punknow = np.zeros((qq))
    pnum = np.zeros((qq))
    pcha = np.zeros((qq))
    for i in range(len(all_pred)):#punknow  pknow 
        if all_pred[i] != open_class:
            pknow[all_pred[i]] = pknow[all_pred[i]] + out_open_all[i, 1, all_pred[i]]#1已知类
            punknow[all_pred[i]] = punknow[all_pred[i]] + out_open_all[i, 0, all_pred[i]]
            pnum[all_pred[i]] = pnum[all_pred[i]] + 1
    for i in range(len(pnum)):
        if pknow[i] != 0 and punknow[i] != 0:
            pknow[i] = pknow[i]/pnum[i]
            punknow[i] = punknow[i]/pnum[i]
    for i in range(len(pnum)):
        pcha[i] = pknow[i] - punknow[i]
    print(pcha,'pcha')
   
    _, predict = torch.max(all_softmax, 1)
    K = all_sm.size(1)//2 +1    
    #print(K,'K') #officehome 16
    
    aff = np.eye(K)[all_pred.cpu().int()]

    #aff = np.eye(K)[all_label.cpu().int()]
    initc = aff.transpose().dot(all_f)   #[21]个聚类中心
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])


   
    labelset = labelset_s  
    labelsets = labelset_s  
    labelsets = labelsets.tolist()
    labelsets.append(len(labelset))
    labelsets = torch.tensor(labelsets).numpy()
    #print(len(initc),'initc')
    print(len(labelset),'labelset')
    
    initck = initc[len(labelset)]
   
    cls_count = np.eye(K)[all_pred1.cpu().int()].sum(axis=0)    

    
    #print(len(initc),len(initc[0]),'initc, labelset')
    dd = cdist(initc[labelset], all_f, args.distance)
    sd = cdist(initc_s[labelset], all_f, args.distance)
    cc = cdist(initc[labelset], initc[labelset], args.distance)
    ds = cdist(initc[labelsets], all_fs, args.distance)
    
    #print(cc,len(cc),len(cc[0]),'cc')   
    ddx = np.argsort(dd, axis=1) 
    sdx = np.argsort(sd, axis=1) 
    dsx = np.argsort(ds, axis=1) 
    ccx = np.argsort(cc, axis=1) 
    
    dd1 = np.sort(dd, axis=1)
    

    ##源域中每类样本数量
    result = []
    arr = all_labels.tolist()
    for i in set(arr):
        result.append(arr.count(i))
    ########################
    
    class_numl_w = np.zeros(len(dd))
    class_numl_c = np.zeros(len(dd))
    class_numl_fc = np.zeros(len(dd))
    cor_numl_fc = np.zeros(len(dd))
    class_numl_un = np.zeros(len(dd))
    num = [[] for i in range(len(dd))]
    
    num_sun = [[] for i in range(len(dd))]
    #print(num_sun,len(num_sun[0]),'num_sun')
    #print(len(labelset),'len(labelset)')
    for i in range(len(dd)):
        for j in range(int(all_f.size(0)/len(labelset)*0.2)): 
        
            if all_pred[ddx[i][j]] != i:
                class_numl_w[i] += 1
            if all_pred[ddx[i][j]] == i:#相邻的正确样本
                class_numl_c[i] += 1

            if all_pred[ddx[i][j]] == len(labelset):
                class_numl_un[i] += 1
                num[i].append(j)
            if all_pred[ddx[i][j]] == i:
                if len(num_sun[i]) == 0:
                    num_sun[i].append(j)
    
    

    avg = int(all_f.size(0)/len(labelset)*0.2) 
    

    for i in range(len(sd)):
        for j in range(int(per_class_fc_num[i])):
            #print(j)
            if all_pred[sdx[i][j]] == i:
                class_numl_fc[i] += 1
        for k in range(int(avg)):
            if all_pred[sdx[i][k]] == i:
                cor_numl_fc[i] += 1
    ###########################3
    class_numl_ds = np.zeros(len(dd))
    class_numl_ds8 = np.zeros(len(dd))
    
    for i in range(len(dd)):
        for j in range(result[i]):
            if all_labels[dsx[i][j]] != i:
                class_numl_ds[i] += 1
        for c in range(int(result[i]*0.8)):   
            if all_labels[dsx[i][c]] != i:
                class_numl_ds8[i] += 1
    
    if cor_fc != []:
        cor_fc = (cor_numl_fc/avg + cor_fc)/2
    else:
        cor_fc = cor_numl_fc/avg
   
    all_feat = []
    clean_feat = []
    clean_la = []
    uninitc = []
    tthres = 7*0.55*(1+step/10000)
    print(tthres,'tthres')

    for i in range(len(dd)):
        if class_numl_w[i] != 0:
            clean_num[i] += 1
    for i in range(len(dd)):
        
        if class_numl_c[i] == avg: 
            
            feat_c = [0 for _ in range(2048)] 
            for k in range(2048):
                feat_c[k] = initc[i][k]
            clean_feat.append(feat_c)
            clean_la.append(i)
    
    print(clean_la,len(clean_la),'len(all_feat),')
    
    index = torch.randperm(len(clean_feat))
    
    for i in range(len(dd)):
        if class_numl_un[i] >= 1:

            for j in range(np.size(unfeat,0)):
                feat_un = [0 for _ in range(2048)]
                lam = np.random.beta(5,1)   
                for k in range(2048):
                    feat_un[k] = lam*initc[i][k] + (1-lam)* unfeat[j][k] 
                    
                all_feat.append(feat_un)
    print(np.size(all_feat,0),'all_feat')
   
    for i in range(len(dd)):
        if class_numl_un[i] >= 1:
            feat_un = [0 for _ in range(2048)]
            for k in range(2048):
                feat_un[k] = initc[i][k]
            uninitc.append(feat_un)

         
    return initc, labelset, all_feat,clean_feat,clean_la,clean_num,cor_fc,uninitc,initck






def evaluate_sourcecentroid(val_loader: DataLoader, model, args: argparse.Namespace, esem):
    model.eval()
    esem.eval()
    #all_output = list()
    #cnt = 0
    start_test = True
    with torch.no_grad():
        for i, data in enumerate(val_loader): #(images, labels)
            img_s, labels = data[0].cuda(), data[1].cuda()
            #images = images.to(device)
            f = model(img_s)
            
            y_1 = esem(f)

            out_open1 = y_1.view(img_s.size(0), 2, -1)
            logits_o1 = F.softmax(out_open1, 2)
            all_soft = logits_o1[:, 1, :]
           
            if(start_test):
                all_f = f.float().cpu()
                all_sm= y_1.float().cpu()
                all_label = labels.float()
                all_softmax = all_soft.float().cpu()
                start_test = False
            else:
                all_f = torch.cat((all_f, f.float().cpu()), 0)#按行拼接
                all_sm = torch.cat((all_sm, y_1.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_softmax = torch.cat((all_softmax, all_soft.float().cpu()), 0)
    
    all_f = (all_f.t() / torch.norm(all_f, p=2, dim=1)).t()#torch.norm按行求2范数  #重要
    _, predict = torch.max(all_softmax, 1)
    K = all_sm.size(1)//2
    #print(K,'K')
    #print(predict,'predict')
    aff = np.eye(K)[all_label.cpu().int()]
    initc = aff.transpose().dot(all_f)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[all_label.cpu().int()].sum(axis=0)                                                                                                                                                                                                   
    #如果对应的类别号是predict，那么转成one-hot的形式
    labelset = np.where(cls_count > 0)  #args.thresholdd
    labelset = labelset[0]

    
    avg_s = [0 for i in labelset]
    mun_s = [0 for i in labelset]#大于平均距离的点都外扩
    in_s = [[] for i in labelset]#内边界
    out_s = [[] for i in labelset]#外边界

    dd = cdist(all_f, initc[labelset], args.distance)
    cc = cdist(initc, initc, args.distance)#[20,20]，源域中心到源域中心的距离
    #print(len(cc),'len(all_label)')
    for i in labelset:
        for j in range(len(all_label)):
            if all_label[j] == i:
                avg_s[i] = avg_s[i] + dd[j][i]
                mun_s[i] = mun_s[i] + 1
                in_s[i].append(dd[j][i])
                
                for m in labelset:
                    if m != i:
                        out_s[m].append(dd[j][m])
        
    
    avg_s1 = torch.tensor(avg_s)/torch.tensor(mun_s)
    #print(avg_s1,'avg_s1')
    
    in_ss = [0 for i in labelset]
    out_ss = [0 for i in labelset]
    
    
   
    for k in range(len(in_s)):
        in_s[k] = np.sort(in_s[k], axis=-1)[::-1]
   
    for i in labelset:
        in_ss[i] = (in_s[i][0] + in_s[i][1] + in_s[i][2] + out_s[i][3] + out_s[i][4])/5
    

    for k in range(len(out_s)):
        out_s[k] = np.sort(out_s[k], axis=-1)[::1] 
    #print(out_s,'out_s')    
    for i in labelset:
        out_ss[i] = (out_s[i][0] + out_s[i][1] + out_s[i][2] + out_s[i][3] + out_s[i][4])/5
    

    return initc, labelset, in_ss, out_ss, avg_s1, cc
