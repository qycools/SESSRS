import csv
import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  


def get_tp_fp_tn_fn(hist):
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    tn = np.diag(hist).sum() - np.diag(hist)
    return tp, fp, tn, fn

def get_Precision(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    precision = tp / (tp + fp)
    return precision

def get_Recall(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    recall = tp / (tp + fn)
    return recall

def get_F1(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = (2.0 * Precision * Recall) / (Precision + Recall)
    return F1

def get_Intersection_over_Union(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    IoU = tp / (tp + fn + fp)
    return IoU

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 


def contrast_metric(gt_dir, pred_dir,sam_p_dir,png, start,end,num_classes,name_classes):
    hist_pred = np.zeros((num_classes, num_classes))
    hist_sam = np.zeros((num_classes, num_classes))
    label = np.array(Image.open(os.path.join(gt_dir,png)))
    pred = np.array(Image.open(os.path.join(pred_dir,png)))
    sam_p = np.array(Image.open(os.path.join(sam_p_dir,png)))

    hist_pred += fast_hist(label.flatten(), pred.flatten(),num_classes) 
    # hist_pred = hist_pred[start:end,start:end]
    hist_sam += fast_hist(label.flatten(), sam_p.flatten(),num_classes) 
    # hist_sam = hist_sam[start:end,start:end]

    IoUs_pred        = per_class_iu(hist_pred)[start:end]
    Pa_Recall_pred   = per_class_PA_Recall(hist_pred)[start:end]
    Precision_pred   = per_class_Precision(hist_pred)[start:end]

    IoUs_sam        = per_class_iu(hist_sam)[start:end]
    Pa_Recall_sam   = per_class_PA_Recall(hist_sam)[start:end]
    Precision_sam   = per_class_Precision(hist_sam)[start:end]


    name_classes = name_classes[start:end]
    for ind_class in range(start,end):
        print(f'===>{name_classes[ind_class]}:\tIou:{round(IoUs_pred[ind_class] * 100, 2)},{round(IoUs_sam[ind_class] * 100, 2)},{round(IoUs_pred[ind_class] * 100-IoUs_sam[ind_class] * 100, 2)};\
               Recall (equal to the PA):{round(Pa_Recall_pred[ind_class] * 100, 2)},{round(Pa_Recall_sam[ind_class] * 100, 2)},{round(Pa_Recall_pred[ind_class] * 100-Pa_Recall_sam[ind_class] * 100, 2)}; \
               Precision:{round(Precision_pred[ind_class] * 100, 2)},{round(Precision_sam[ind_class] * 100, 2)},{round(Precision_pred[ind_class] * 100-Precision_sam[ind_class] * 100, 2)}')
       
    print(f'===> mIoU:{round(np.nanmean(IoUs_pred) * 100, 2)},{round(np.nanmean(IoUs_sam) * 100, 2)},{round(np.nanmean(IoUs_pred) * 100-np.nanmean(IoUs_sam) * 100, 2)}; \
        mPA_Recall:{round(np.nanmean(Pa_Recall_pred) * 100, 2)},{round(np.nanmean(Pa_Recall_sam) * 100, 2)},{round(np.nanmean(Pa_Recall_pred) * 100-np.nanmean(Pa_Recall_sam) * 100, 2)};\
        Accuracy:{round(per_Accuracy(hist_pred) * 100, 2)},{round(per_Accuracy(hist_sam) * 100, 2)},{round(per_Accuracy(hist_pred) * 100-per_Accuracy(hist_sam) * 100, 2)}\nGet miou done.\n\n')

def contrast_metric2(gt_dir, sam_p_dir,sam_p_dir2,png, start,end,num_classes,name_classes):
    # hist_pred = np.zeros((num_classes, num_classes))
    hist_sam = np.zeros((num_classes, num_classes))
    label = np.array(Image.open(os.path.join(gt_dir,png)))
    # pred = np.array(Image.open(os.path.join(pred_dir,png)))
    sam_p = np.array(Image.open(os.path.join(sam_p_dir,png)))
    sam_p2 = np.array(Image.open(os.path.join(sam_p_dir2,png)))

    # hist_pred += fast_hist(label.flatten(), pred.flatten(),num_classes) 
    hist_sam += fast_hist(label.flatten(), sam_p.flatten(),num_classes) 
    hist_sam2 += fast_hist(label.flatten(), sam_p2.flatten(),num_classes) 

    # hist_pred = hist_pred[start:end,start:end]
    hist_sam = hist_sam[start:end,start:end]
    hist_sam2 = hist_sam2[start:end,start:end]

    # IoUs_pred        = per_class_iu(hist_pred)
    # Pa_Recall_pred   = per_class_PA_Recall(hist_pred)
    # Precision_pred   = per_class_Precision(hist_pred)

    IoUs_sam        = per_class_iu(hist_sam)
    Pa_Recall_sam   = per_class_PA_Recall(hist_sam)
    Precision_sam   = per_class_Precision(hist_sam)

    IoUs_sam2       = per_class_iu(hist_sam2)
    Pa_Recall_sam2  = per_class_PA_Recall(hist_sam2)
    Precision_sam2  = per_class_Precision(hist_sam2)

    name_classes = name_classes[start:end]
    for ind_class in range(start,end):
        print(f'===>{name_classes[ind_class]}:\tIou:{round(IoUs_sam[ind_class] * 100, 2)},{round(IoUs_sam2[ind_class] * 100, 2)},{round(IoUs_sam[ind_class] * 100-IoUs_sam2[ind_class] * 100, 2)};\
               Recall (equal to the PA):{round(Pa_Recall_sam[ind_class] * 100, 2)},{round(Pa_Recall_sam2[ind_class] * 100, 2)},{round(Pa_Recall_sam[ind_class] * 100-Pa_Recall_sam2[ind_class] * 100, 2)}; \
               Precision:{round(Precision_sam[ind_class] * 100, 2)},{round(Precision_sam2[ind_class] * 100, 2)},{round(Precision_sam[ind_class] * 100-Precision_sam2[ind_class] * 100, 2)}')
       
    print(f'===> mIoU:{round(np.nanmean(IoUs_sam) * 100, 2)},{round(np.nanmean(IoUs_sam2) * 100, 2)},{round(np.nanmean(IoUs_sam) * 100-np.nanmean(IoUs_sam2) * 100, 2)}; \
        mPA_Recall:{round(np.nanmean(Pa_Recall_sam) * 100, 2)},{round(np.nanmean(Pa_Recall_sam2) * 100, 2)},{round(np.nanmean(Pa_Recall_sam) * 100-np.nanmean(Pa_Recall_sam2) * 100, 2)};\
        Accuracy:{round(per_Accuracy(hist_sam) * 100, 2)},{round(per_Accuracy(hist_sam2) * 100, 2)},{round(per_Accuracy(hist_sam) * 100-per_Accuracy(hist_sam2) * 100, 2)}\nGet miou done.\n\n')

def compute_mIoU_all(gt_dir, pred_dir, png_name_list, num_classes, name_classes,start,end):  
  
    hist = np.zeros((num_classes, num_classes))

    gt_imgs     = [join(gt_dir, x) for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x) for x in png_name_list]  

    for ind in range(len(gt_imgs)): 

        pred = np.array(Image.open(pred_imgs[ind]))  
        label = np.array(Image.open(gt_imgs[ind]))  

        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
   
    IoUs        = get_Intersection_over_Union(hist)[start:end]
    PA_Recall   = get_Recall(hist)[start:end]
    Precision   = get_Precision(hist)[start:end]
    F1          = get_F1(hist)[start:end]
    name_classes = name_classes[start:end]  

    for ind_class in range(len(name_classes)):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
            + '; F1-' + str(round(F1[ind_class] * 100, 2))\
            + '; Recall-' + str(round(PA_Recall[ind_class] * 100, 2))\
                + '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + \
          '; mF1: ' + str(round(np.nanmean(F1) * 100, 2)) + \
          '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + \
            '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
     
def print_hist(gt_dir, pred_dir, png_name_list, name_classes):  
    
    num_classes = len(name_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs     = [join(gt_dir, x) for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x) for x in png_name_list]  

    for ind in range(len(gt_imgs)): 

        pred = np.array(Image.open(pred_imgs[ind]))  
        label = np.array(Image.open(gt_imgs[ind]))  

        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
    
    pred_name = pred_dir.split('/')[-2]
    with open(os.path.join('csv_demo',f'{pred_name}.csv'), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
 
def get_miou_all_main(pred_p_dir,gt_dir,name_classes,image_ids,start,end):
 
    num_classes     = len(name_classes)
    print("Get miou.")
    compute_mIoU_all(gt_dir, pred_p_dir, image_ids, num_classes, name_classes,start,end)  # 执行计算mIoU的函数
    print("Get miou done.")
    print("\n\n")


