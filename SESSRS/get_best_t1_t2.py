import sys
import os
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from typing import Any, ItemsView
from copy import deepcopy
import pycocotools.mask as mask_utils
from pycocotools.mask import *
import json
import multiprocessing
from get_miou import get_miou_all_main
from scipy.ndimage import label, find_objects
import argparse
import re

class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.detach().cpu().numpy()

def segmentation_2_instance(label_image):  
    """  
    获取语义分割标签图中每个标签的边界框。  
  
    参数:  
    label_image (np.ndarray): 语义分割标签图，应为整数类型。  
    min_area (int): 忽略面积小于此值的标签。  
  
    返回:  
    dict: 键为标签值，值为边界框列表 [(x, y, w, h), ...]  

    注意：
    不会获取标签0 的info
    """  
    
    unique_labels = np.unique(label_image)  
    unique_labels = unique_labels[unique_labels!=0]
    bounding_boxes = {label: [] for label in unique_labels}
    
    data = MaskData()
    for label in unique_labels:  

        mask = (label_image == label)  
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        sizes = stats[:,4:5][1:]
        bounding_boxes = stats[:, :-1][1:]
        instance_labels = []
        for i in range(1,num_labels):
            instance_label = (labels==i)
            instance_labels.append(instance_label)
        instance_labels = np.array(instance_labels)

        batch_data = MaskData(
            masks = instance_labels,
            bbox = bounding_boxes,
            semantic = np.full(sizes.shape,label),
            sizes = sizes
        ) 

        data.cat(batch_data)
    data['rles'] = [singleMask2rle(mask.astype(np.uint8)) for mask in data['masks']]
    data_list = []
    for idx in range(len(data["masks"])):
            
        ann = {"rles": data["rles"][idx]}
        data_list.append(ann)

    return data_list

def singleMask2rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle

def sessrs(path):

    with open(os.path.join(pre_p_info_path,path[:-4]+'.json'), 'r') as file:  
        label = json.load(file)

    with open(os.path.join(sam_label_info_path,path[:-4]+'.json'), 'r') as file:  
        mask = json.load(file)

    label_segmentation = np.array(Image.open(os.path.join(p_dir,path)))
    H,W = label_segmentation.shape[0],label_segmentation.shape[1]   
    Ak={}
    for k in range(num_classes):
        Ak[f'label{k}'] = []
    
    for m in mask:
        mask_l_segmentation = mask_utils.decode(m['rles']).astype(np.bool_)
        areas        = m['area']
        counts      = np.bincount(label_segmentation[mask_l_segmentation])
        if counts.size == 0:
            m['semantic'] = None
            m['e_counts_ratio'] = None
            continue
        else:
            most_common_element = np.argmax(counts)
            e_counts            = counts[most_common_element]

            if e_counts/areas > args.fix_t1[f'{most_common_element}']:
                
                m['semantic'] = most_common_element
                m['e_counts_ratio'] = round(e_counts/areas,3)
            else:
                m['semantic'] = None
                m['e_counts_ratio'] = None

    mask = sorted(mask,key=lambda x: x['area'],reverse=True)
    for i in range(len(mask)-1):
        mask_i_sementic = mask[i]['semantic']
        if mask_i_sementic in args.modify_category:
            for j in range(i+1,len(mask)): 
                mask_j_sementic = mask[j]['semantic']
                if mask_i_sementic != mask_j_sementic and mask_j_sementic is not None:
                    if area(merge([mask[i]['rles'],mask[j]['rles']],intersect=True))/mask[j]['area']>args.fix_t2[f'{mask_i_sementic}'] and mask[i]['e_counts_ratio']<mask[j]['e_counts_ratio']:
                        mask[i]['rles'] = deintersect(mask[i]['rles'],mask[j]['rles'])
                        mask[i]['area'] = area(mask[i]['rles'])
                else:
                    continue
        else:
            continue

    for m in mask:
        semantic = m['semantic']
        if semantic is not None:
            Ak[f'label{semantic}'].append(m)
              
    
    
    Pk={}
    for k in range(num_classes):
        Pk[f'label{k}'] = []
    
    for l in label:
        for k in range(num_classes):
            if l['semantic'] == str(k):
                Pk[f'label{k}'].append(l)

    # sessrs_Pk:包含合并sessrs后某一类型所有Pk
    sessrs_Pk = {}
    # sessrsed_Pk:包含所有被sessrs处理过的pk
    # sessrs_Pk和sessrsed_Pk的区别在于sessrs_Pk是某种地物类型全部合并后的结果，sessrsed_Pk是分散的
    # 初始化sessrsed_Pk
    
    for k in args.modify_category:
        flag = 0
        t1 = args.fix_t1[f'{k}']
        t2 = args.fix_t2[f'{k}']
        for p_k in Pk[f'label{k}']:
            tmp =[]
            for mask_S in Ak[f'label{k}']:

                if area(mask_S['rles'])>200000:
                    tmp =[]
                    continue

                Os = area(merge([p_k['rles'],mask_S['rles']],intersect=True))/mask_S['area']
                Op = area(merge([p_k['rles'],mask_S['rles']],intersect=True))/p_k['size']
                if Os>t1 or Op>t2:
                    tmp.append(mask_S['rles'])

            if len(tmp)>1 and len(tmp)<5 :

                tmp_rles = merge(tmp)
                if area(merge([tmp_rles,p_k['rles']],intersect=True))/area(p_k['rles']) > t2:
                
                    p_k['rles'] = tmp_rles
                    Pk[f'label{k}'][flag] = p_k

            elif len(tmp)==1 and area(tmp[0]) >t2*area(p_k['rles']) and area(tmp[0])*t1 <area(p_k['rles']):
                p_k['rles'] = tmp[0]
                Pk[f'label{k}'][flag] = p_k
            else:
                pass

            flag = flag + 1

        if len(Pk[f'label{k}']) == 0:
            sessrs_Pk[f'{k}'] = None
        
        else:
            Pk_rles = [pk['rles'] for pk in Pk[f'label{k}']]
            sessrs_Pk[f'{k}'] = merge(Pk_rles)

    intersect_list=[]
    sessrs_index = args.modify_category
    for i in range(len(sessrs_index)):
        if sessrs_Pk[f'{sessrs_index[i]}'] is None:
            continue
        for j in range(i + 1, len(sessrs_index)):
            if sessrs_Pk[f'{sessrs_index[j]}'] is None:
                continue

            intersect = merge([sessrs_Pk[f'{sessrs_index[i]}'],sessrs_Pk[f'{sessrs_index[j]}']],intersect=True)
            if area(intersect) ==0:
                continue
            intersect = decode(intersect)
            seg_ins_intersect = segmentation_2_instance(intersect)
            for inter in seg_ins_intersect:
                max_i_iou = 0
                max_j_iou = 0

                for pki in Pk[f'label{sessrs_index[i]}']:
                    is_iou =  iou([inter['rles']],[pki['rles']],[0])[0][0]
                    if is_iou > max_i_iou:
                        max_i_iou = is_iou

                for pkj in Pk[f'label{sessrs_index[j]}']:
                    is_iou =  iou([inter['rles']],[pkj['rles']],[0])[0][0]
                    if is_iou > max_j_iou:
                        max_j_iou = is_iou  
                
                if max_i_iou > max_j_iou:
                    inter['semantic'] = sessrs_index[i] 
                else:
                    inter['semantic'] = sessrs_index[j]
                
                intersect_list.append(inter)

    image = np.ones([H,W],dtype=np.uint8)*255
    flag = np.ones([H,W],dtype=np.uint8)

    for inter in intersect_list:
        semantic = int(inter['semantic'])
        seg = decode(inter['rles'])
        image[seg==1] = (semantic*seg*flag)[seg==1]
        flag[flag&seg==1] = 0
    
    for k in args.modify_category:
        # seg_flag = 0
        if sessrs_Pk[f'{k}'] is not None:
            seg = decode(sessrs_Pk[f'{k}'])
            image[flag*seg==1] = (k*seg*flag)[flag*seg==1]
        else:
            continue

    label_segmentation_modify = modify(label_segmentation,args.modify_category)    
    image[image==255] = label_segmentation_modify[image==255]
    image = Image.fromarray(np.uint8(image),'P')
    image.putpalette(palette)
    image.save(os.path.join(sessrs_path,path))

def deintersect(mask1,mask2):
    '''
    mask1去除mask1与mask2的交集部分
    '''
    if isinstance(mask1, dict) and 'size' in mask1 and 'counts' in mask1:
        if isinstance(mask2, dict) and 'size' in mask2 and 'counts' in mask2:
            
            mask1_binary = decode(mask1)
            mask2_binary = decode(mask2)
            if np.shape(mask1_binary) == np.shape(mask2_binary):
            # 从mask2中去除与mask1重叠的部分
                mask1_modified_binary = mask1_binary & ~mask2_binary
                return encode(mask1_modified_binary)
            else:
                raise ValueError('deintersect: mask1 and mask2 must be of the same shape')
    
    elif isinstance(mask1, np.ndarray) and isinstance(mask2, np.ndarray):
        if np.shape(mask1) == np.shape(mask2):
            # 从mask2中去除与mask1重叠的部分1
            mask1_modified_binary = mask1 & ~mask2
            return mask1_modified_binary
        else:
            raise ValueError('deintersect: mask1 and mask2 must be of the same shape')
    
    else:
        raise ValueError('deintersect: mask1 and mask2 must be of the same type')

def modify(orig_mask_array,modify_category_list):

    modified_category_list=[]
    for category in modify_category_list:
        orig_mask_array_copy = orig_mask_array.copy()
        orig_mask_array_copy[orig_mask_array_copy==category] = 255
        modified_category = modify_b_w(orig_mask_array_copy,category)
        modified_category_list.append(modified_category)
    
    orig_mask_array_copy = orig_mask_array.copy()
    for category,modified_category in zip(modify_category_list,modified_category_list):
        orig_mask_array_copy[orig_mask_array==category] = modified_category[orig_mask_array==category]
    return  orig_mask_array_copy

def modify_b_w(array,category):
# 用0值填充有问题
    # 用0值填充数组周围，使其变为[1026, 1026]
    padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=254)

    # 找到值为8的连通区域
    labeled_array, num_features = label(padded_array == 255)

    # 获取连通区域的边界
    slices = find_objects(labeled_array)

    for i, slc in enumerate(slices, start=1):
        mask = (labeled_array[slc] == i)
        
        # 扩展掩码，用于检测边界
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=254)
        extended_region = np.pad(padded_array[slc], pad_width=1, mode='constant', constant_values=254)
        
        # 获取边界掩码
        border_mask = np.zeros_like(padded_mask)
        border_mask[1:-1, 1:-1] = (
            padded_mask[1:-1, 1:-1] &
            ((~padded_mask[:-2, 1:-1]) | (~padded_mask[2:, 1:-1]) |
            (~padded_mask[1:-1, :-2]) | (~padded_mask[1:-1, 2:]) |
            (~padded_mask[:-2, :-2]) | (~padded_mask[2:, 2:]) |
            (~padded_mask[:-2, 2:]) | (~padded_mask[2:, :-2]))
        )
        
        border_indices = np.argwhere(border_mask == 1)
        list2 = []
        for idx in border_indices:
            x, y = idx
            neighbors = extended_region[x-1:x+2, y-1:y+2].flatten()
            neighbors = neighbors[neighbors != 255]
            neighbors = neighbors[neighbors != 254]  
            list2.extend(neighbors)

        most_common_value = category
        if len(list2) > 0:
            most_common_value = np.bincount(list2).argmax()

        padded_array[slc][mask] = most_common_value

    # 去除周围填充0值，使数组重新变为[1024, 1024]大小
    result_array = padded_array[1:-1, 1:-1]
    return result_array

def multipool(function, args,processes=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool(processes=processes)
    with tqdm(total=len(args)) as pbar:
        for result in pool.imap_unordered(function, args):
            pbar.update(1)
    pool.close()
    pool.join()

def get_best_sessrs(cpu_num):
    
    multipool(sessrs,img_paths,cpu_num)
    get_miou_all_main(p_dir,gt_dir,name_classes,img_paths,class_start,class_end)
    get_miou_all_main(sessrs_path,gt_dir,name_classes,img_paths,class_start,class_end)
    
def run_category_t1_t2_txt(txt_file):
    groups_of_four = []

    with open(txt_file, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if i + 3 < len(lines):

                group = [lines[i].strip(), lines[i+1].strip(), lines[i+2].strip(), lines[i+3].strip()]
                groups_of_four.append(group)
                i += 7
            else:
                break
    
    return groups_of_four

def test_category(test_file,reverse=True):
    # 测试t1,t2对精度的影响  
    with open(test_file, 'r') as f:  
        # 保存原来的sys.stdout  
        txts = f.readlines()
    
    metrics = []
    for line in txts:
        if 't1' in line :
            line =line.strip()
            pairs = line.split(',')
            result={}
            for pair in pairs:
                key, value = pair.split('=')
                result[key] = float(value)
            metrics.append(result)
        if line.startswith('===> mIoU:'):
            numbers = re.findall(r'-?\d+\.\d+|-?\d+', line) 
            float_numbers = [float(num) for num in numbers]
            metrics.append(float_numbers)

    metrics_order = []
    i = 0
    while i < len(metrics):
        result = metrics[i]
        result['iou'] = metrics[i+1][0]
        metrics_order.append(result)
        i = i+2

    metrics_order = sorted(metrics_order,key = lambda x:x['iou'],reverse = reverse)
    return metrics_order

def test_model(test_dir,startwith):
    # 测试model下每个类别的t1,t2对结果的影响
    # test_files = sorted([ test_file for test_file in os.listdir(os.path.join(test_dir,args.write_dir))])
    test_files =  sorted([ test_file for test_file in os.listdir(os.path.join(test_dir,args.write_dir)) \
                          if test_file.startswith(startwith)and test_file.endswith('.txt')])
    if len(test_files) == 0:
        return
    best_category_t1 = {}
    best_category_t2 = {}

    for index in range(num_classes):
        best_category_t1[f'{index}'] = 1.0
        best_category_t2[f'{index}'] = 1.0

    modify_category = []
    flag = 0
    for test_file in test_files:
        best_category_t1[f'{flag}'] = test_category(os.path.join(test_dir,args.write_dir,test_file))[0]['t1']
        best_category_t2[f'{flag}'] = test_category(os.path.join(test_dir,args.write_dir,test_file))[0]['t2']
        if best_category_t1[f'{flag}'] !=1 and best_category_t2[f'{flag}'] !=1:
            modify_category.append(flag)
        flag = flag+1

    print(test_dir)
    print('args.fix_t1 = '+str( best_category_t1))
    print('args.fix_t2 = '+str( best_category_t2))
    print('args.modify_category = '+str(modify_category))
    print('\n\n')

def test_all(dataset_dir,text_file):
    # 测试dataset_dir所有的model的sessrs遍历结果
    startwith = text_file.split('/')[-1].split('_t2')[0]
    f = open(text_file,'w')
    sys.stdout = f
    test_dirs = sorted([os.path.join(dataset_dir,test_dir) for test_dir in os.listdir(dataset_dir) \
                        if os.path.isdir(os.path.join(dataset_dir,test_dir)) and 'pre_p' in os.listdir(os.path.join(dataset_dir,test_dir)) ])
    for test_dir in test_dirs:
        test_model(test_dir,startwith)
    f.close()

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-d", "--dataset", default='Urban',choices=['Urban','potsdam'])
    arg("-m", "--modify_type", default='modify',choices=['modify','no_modify'])
    arg("-s", "--sam_type", default='sam',choices=['sam','samhq','ssam'])
    arg("-w", "--write_dir",default='sessrs_write')
    arg("-t", "--threshold_mode", default=0.7)
    arg("-n", "--multipool_nums", default=64,help='The number of multiple processes')
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    all_file_start = {'modify':'modify_sessrs','no_modify':'sessrs'}

    figresult_dataset_dir_dict = {'potsdam':'../GeoSeg/fig_results/potsdam',
                                'Urban': '../GeoSeg/fig_results/Urban'}
    
    dataset_dir_dict = {'potsdam':'../GeoSeg/data/potsdam',
                        'Urban': '../GeoSeg/data/Urban'}
  
    gt_dir_end = {'no_modify':'/val/masks','modify':'/val/modify_masks'}

    name_classes_dict  = {'potsdam': ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter'],
                        'Urban':['background', 'building', 'road', 'water', 'barren', 'forest','agricultural','other']}
    
    class_start_dict = {'potsdam':0,'Urban':0}
    class_end_dict = {'potsdam':6,'Urban':7}

    palette_dict = {'potsdam':[255, 255, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0, 255, 204, 0, 255, 0, 0],
                    'Urban':[255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 255,159, 129, 183, 0, 255, 0, 255, 195, 128,0,0,0]}
    
    if args.threshold_mode == 'all':
        threshold_dir = '/sessrs_contrast_all'
        threshold = 0.99
    else:
        threshold_dir = f'/sessrs_contrast_{args.threshold_mode}'
        threshold = float(args.threshold_mode)
    dataset_dir         = dataset_dir_dict[args.dataset]
    all_file_name       = f'{all_file_start[args.modify_type]}_{args.sam_type}_t1_t2_all.txt'
    fig_results_dir     = figresult_dataset_dir_dict[args.dataset]
    os.makedirs(fig_results_dir + threshold_dir, exist_ok=True)
    all_file            = fig_results_dir + threshold_dir + '/'+ all_file_name
    sam_label_info_path  = dataset_dir + f'/sam_label_info/{args.sam_type}'

    gt_dir       = dataset_dir+gt_dir_end[args.modify_type]
    name_classes = name_classes_dict[args.dataset]
    num_classes  = len(name_classes)
    val_txt      = dataset_dir+'/val.txt'
    class_start  = class_start_dict[args.dataset]
    class_end    = class_end_dict[args.dataset]
    palette      = palette_dict[args.dataset]

    test_all(fig_results_dir,all_file)

    groups_of_four      = run_category_t1_t2_txt(all_file)
    contrast_file_name  = all_file_name.split('t1')[0]+'ps_contrast.txt'
    contrast_file       = fig_results_dir + threshold_dir + '/' + contrast_file_name

    f = open(contrast_file,'w')
    sys.stdout = f
    for group in groups_of_four:
        fig_results         = group[0]
        print(fig_results+'\n')
        exec(group[1])                    
        exec(group[2])                    
        exec(group[3])
        
        args.modify_category = [int(key) for key, value in args.fix_t1.items() if value <= threshold]   

        p_dir               = fig_results+'/pre_p'
        pre_p_info_path     = fig_results+'/pre_p_info'
        sessrs_path           = fig_results+'/se_mask'
        img_paths = sorted(os.listdir(p_dir))
        get_best_sessrs(args.multipool_nums)

    f.close()