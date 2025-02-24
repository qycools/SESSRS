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

def segmentation_2_instance_json(image_path):  
    """  
    获取语义分割标签图中每个标签的边界框。  
  
    参数:  
    label_image (np.ndarray): 语义分割标签图，应为整数类型。  
    min_area (int): 忽略面积小于此值的标签。  
  
    返回:  
    dict: 键为标签值，值为边界框列表 [(x, y, w, h), ...]  

    注意：
    不会获取标签0 的info
    7.12修改会获取
    """  
    # 获取标签的唯一值  
    label_image = np.array(Image.open(os.path.join(p_dir,image_path)))
    unique_labels = np.unique(label_image)  
    # unique_labels = unique_labels[unique_labels!=0]
    bounding_boxes = {label: [] for label in unique_labels}
    
    data = MaskData()
    for label in unique_labels:  
        # 创建标签的掩码  
        mask = (label_image == label)  
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # 生成第一个是背景像素，所有连通域得从1开始
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
            
            ann = {
                'semantic':str(data["semantic"][idx][0]),
                "rles": data["rles"][idx],
                "bbox": data["bbox"][idx].tolist(),
                "size": int(data["sizes"][idx][0])
                    
            }
            data_list.append(ann)

    
    json_data = json.dumps(data_list, indent=4, ensure_ascii=False)  
    # np.savetxt( os.path.join(os.path.join(g_lbl_josn_file,f'All_prompt'),zr_img.split('.')[0]+'.txt'),np.array(sam_info))
    with open(os.path.join(pre_p_info_path,image_path.split('.')[0]+'.json'),'w') as f:  
        f.write(json_data)

    return data

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
    # 获取标签的唯一值  
    
    unique_labels = np.unique(label_image)  
    unique_labels = unique_labels[unique_labels!=0]
    bounding_boxes = {label: [] for label in unique_labels}
    
    data = MaskData()
    for label in unique_labels:  
        # 创建标签的掩码  
        mask = (label_image == label)  
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # 生成第一个是背景像素，所有连通域得从1开始
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
    # orig_mask_array = np.array(Image.open(img_path,img_path))
    modified_category_list=[]
    for category in modify_category_list:
        orig_mask_array_copy = orig_mask_array.copy()
        orig_mask_array_copy[orig_mask_array_copy==category] = 255
        modified_category = modify_b_w(orig_mask_array_copy,category)
        modified_category_list.append(modified_category)
    
    # bug1:之前的代码有点问题，因为orig_mask_array在变化，
    # 改代码就存在问题orig_mask_array[orig_mask_array==category] = modified_category[orig_mask_array==category]
    orig_mask_array_copy = orig_mask_array.copy()
    for category,modified_category in zip(modify_category_list,modified_category_list):
        orig_mask_array_copy[orig_mask_array==category] = modified_category[orig_mask_array==category]
    return  orig_mask_array_copy

def modify_b_w(array,category):

    # 用254填充数组周围，使其变为[1026, 1026]
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

    # 去除周围填充值，使数组重新变为[1024, 1024]大小
    result_array = padded_array[1:-1, 1:-1]
    return result_array

def sessrs(path):
    # 与sers_all相比 改动在改9.10 那儿，限制了len(tmp)的数量
    #  并且sers_all原来的modify 函数有问题，现已经修改
    # num_classes = 8
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

            # 相当于(Sl ∩ merge(Pk))/S > t1
            if e_counts/areas > args.fix_t1[f'{most_common_element}']:
                
                m['semantic'] = most_common_element
                m['e_counts_ratio'] = round(e_counts/areas,3)
                # Ak[f'label{most_common_element}'].append(m)
            else:
                m['semantic'] = None
                m['e_counts_ratio'] = None

    # 改动7:处理一下重叠的不同地物类型的mask
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

                # 改6：限制大面积改动
                if area(mask_S['rles'])>200000:
                    tmp =[]
                    continue

                Os = area(merge([p_k['rles'],mask_S['rles']],intersect=True))/mask_S['area']
                Op = area(merge([p_k['rles'],mask_S['rles']],intersect=True))/p_k['size']
                if Os>t1 or Op>t2:
                    tmp.append(mask_S['rles'])
            # 改1：不改的话即使满足Os>t1，还要满足Op>t2。
            # if len(tmp)>0:
            # 9.10 改 and len(tmp)<5
            if len(tmp)>1 and len(tmp)<10 :
                # 相当于(Pk ∩ merge(S))/Pk > t2
                tmp_rles = merge(tmp)
                # 改4：不再使用原始面积判断，而是用相交区域比例
                if area(merge([tmp_rles,p_k['rles']],intersect=True))/area(p_k['rles']) > t2:
                
                    p_k['rles'] = tmp_rles
                    Pk[f'label{k}'][flag] = p_k
            # 改5：当len(tmp)==1时，需要Mask_S和p_k差不多大，面积得限制
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

    # 改3：当有重叠区域根据iou判断属于那种类型，而不单纯的根据类型顺序赋值
    intersect_list=[]
    sessrs_index = args.modify_category
    for i in range(len(sessrs_index)):
        if sessrs_Pk[f'{sessrs_index[i]}'] is None:
            continue
        for j in range(i + 1, len(sessrs_index)):
            if sessrs_Pk[f'{sessrs_index[j]}'] is None:
                continue
            # 调用函数进行比较
            intersect = merge([sessrs_Pk[f'{sessrs_index[i]}'],sessrs_Pk[f'{sessrs_index[j]}']],intersect=True)
            if area(intersect) ==0:
                continue
            intersect = decode(intersect)
            seg_ins_intersect = segmentation_2_instance(intersect)
            # seg_ins_intersect = [inter for inter in segmentation_2_instance(intersect) if area(inter['rles']) > 50]
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
    # flag 为0代表该处有重叠区域
    flag = np.ones([H,W],dtype=np.uint8)
    
    # 先绘制重叠区域
    for inter in intersect_list:
        semantic = int(inter['semantic'])
        seg = decode(inter['rles'])
        image[seg==1] = (semantic*seg*flag)[seg==1]
        flag[flag&seg==1] = 0
    

    for k in args.modify_category:
        # seg_flag = 0
        if sessrs_Pk[f'{k}'] is not None:
            seg = decode(sessrs_Pk[f'{k}'])
            # bug2: seg==1 --> flag*seg == 1,原先的写法导致重叠部分的代码无效
            image[flag*seg==1] = (k*seg*flag)[flag*seg==1]
        else:
            continue
    

    # 最后将原图中未改动区域绘制
    # 改进2：原来直接用元素0填充被删除的类型区域，现在根据被删除类型区域周围类型元素数量确定填充类型
    label_segmentation_modify = modify(label_segmentation,args.modify_category)    
    image[image==255] = label_segmentation_modify[image==255]
    # image[image==num_classes] = 0
    image = Image.fromarray(np.uint8(image),'P')
    image.putpalette(palette)
    image.save(os.path.join(sessrs_path,path))

def multipool(function, args,processes=72):
    pool = multiprocessing.Pool(processes=processes)
    # results = pool.imap_unordered(function, args)
    with tqdm(total=len(args)) as pbar:
    # 使用imap_unordered来执行任务并迭代结果
        # for result in pool.imap_unordered(starmap(function, args)):
        for result in pool.imap_unordered(function, args):
            # 这里可以处理任务结果
            # 在任务完成时更新进度条
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()

def test_t1_t2_2():

    modify_category_range = []
    for i in range(class_start,class_end):
            modify_category_range.append(i)
    
    for i in modify_category_range:
        args.modify_category = [i] 
        write_txt = write_starts+f'_{i}.txt'
        f = open(write_txt,'w')
        sys.stdout = f
        print('t1=1,t2=1')

        get_miou_all_main(p_dir,gt_dir,name_classes,img_paths,class_start,class_end)

        for t1 in [0.5,0.6,0.7,0.8,0.9]:
            for t2 in [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
                print(f't1={t1},t2={t2}')
                args.fix_t1 = {}
                args.fix_t2 = {}
                for j in range(num_classes):
                    args.fix_t1[str(j)] = 1
                    args.fix_t2[str(j)] = 1
                args.fix_t1[str(i)] = t1
                args.fix_t2[str(i)] = t2

                multipool(sessrs,img_paths,args.multipool_nums)
                get_miou_all_main(sessrs_path,gt_dir,name_classes,img_paths,class_start,class_end)
        f.close()

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-f", "--fig_results", default='../GeoSeg/fig_results/Urban/unetformer')
    arg("-m", "--modify_type", default='modify',choices=['modify',None],help = "Whether to use the modified data set ,only Urban")
    arg("-s", "--sam_type", default='sam',choices=['sam','samhq','ssam'])
    arg("-n", "--multipool_nums", default=64,help='The number of multiple processes')
    return parser.parse_args()
    
def makedirs():
    os.makedirs(sessrs_path,exist_ok=True)
    os.makedirs(pre_p_info_path,exist_ok=True)
    os.makedirs(write_dir, exist_ok=True)
    if len(os.listdir(pre_p_info_path)) != len(img_paths):
        multipool(segmentation_2_instance_json,img_paths)
    
if __name__ == "__main__":

    write_file_start = {'modify':'modify_sessrs','no_modify':'modify5'}

    dataset_dir_dict = {'potsdam':'../GeoSeg/data/potsdam',
                        'Urban': '../GeoSeg/data/Urban'}
    
    gt_dir_end = {'no_modify':'/val/masks','modify':'/val/modify_masks'}

    name_classes_dict  = {'potsdam': ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter'],
                        'Urban':['background', 'building', 'road', 'water', 'barren', 'forest','agricultural','other']}
    
    class_start_dict = {'potsdam':0,'Urban':0}
    class_end_dict = {'potsdam':6,'Urban':7}

    palette_dict = {'potsdam':[255, 255, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0, 255, 204, 0, 255, 0, 0],
                    'Urban':[255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 255,159, 129, 183, 0, 255, 0, 255, 195, 128,0,0,0]}

    args                   = get_args()
    fig_results            = args.fig_results
    dataset                = args.fig_results.split('/')[-2]

    p_dir                  = fig_results+'/pre_p'
    pre_p_info_path        = fig_results+'/pre_p_info'
    sessrs_path            = fig_results+'/se_mask'
    palette                = palette_dict[dataset]
    name_classes           = name_classes_dict[dataset]
    num_classes            = len(name_classes)
    dataset_dir            = dataset_dir_dict[dataset]
    class_start            = class_start_dict[dataset]
    class_end              = class_end_dict[dataset]
    sam_label_info_path    = dataset_dir + f'/sam_label_info/{args.sam_type}'
    write_dir              = fig_results+'/sessrs_write'
    write_starts           = write_dir+f'/{write_file_start[args.modify_type]}_{args.sam_type}_t1_t2'
    gt_dir                 = dataset_dir+gt_dir_end[args.modify_type]

    img_paths = sorted(os.listdir(p_dir))
    
    makedirs()
    test_t1_t2_2()