
import torch
import numpy as np
from torchvision import transforms
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
from .automatic_mask_generator import SemanticSamAutomaticMaskGenerator
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

def get_sam_label(model, image,level,text_size,box_nms =0.7,min_mask_region_area=10, pred_iou_thresh=0.88,stability_score_thresh=0.92):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)
    size = image_ori.size
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    mask_generator = SemanticSamAutomaticMaskGenerator(model,
            points_per_side=32,
            points_per_batch=200,
            pred_iou_thresh=pred_iou_thresh,
            box_nms_thresh=box_nms,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            level=level,
        )

    im = mask_generator.generate(images)

    if len(level) == 1:
        label_RGB,label_P = show_demo2(size,im)
    else:
        label_RGB,label_P = show_demo(size,im)
    label = Image.fromarray(label_RGB.astype('uint8')).convert('RGB')
    iimage_ori =Image.fromarray(image_ori.astype('uint8')).convert('RGB')
    blend = Image.blend(iimage_ori,label,0.35)
    label_P = label_P.astype(np.uint16)
    # label_P = Image.fromarray(np.squeeze(label_P).astype('uint8')).convert('P')
    return label,blend,label_P

def get_sam_json(model, image,level,text_size,box_nms =0.7,min_mask_region_area=10, pred_iou_thresh=0.88,stability_score_thresh=0.92):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    mask_generator = SemanticSamAutomaticMaskGenerator(model,
            points_per_side=32,
            points_per_batch=200,
            pred_iou_thresh=pred_iou_thresh,
            box_nms_thresh=box_nms,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            level=level,
        )

    data = mask_generator.generate2(images)
   
    return data

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def unique3D(array):
    uni_arrays =[]
    for i in range(array.shape[0]):
        array_w = np.unique(array[i] ,axis=0)
        uni_arrays.extend(array_w)
    uni_arrays = np.array(uni_arrays)
    unique3d = np.unique(uni_arrays,axis = 0)
    return unique3d

def get_category(images,unique3d):
    category = np.zeros((images.shape[0],images.shape[1]))
    flag = 1
    for item in unique3d:
        mask1 = np.where(images==item,1,0)[:,:,0]
        mask2 = np.where(images==item,1,0)[:,:,1]
        mask3 = np.where(images==item,1,0)[:,:,2]
        mask = mask1*mask2*mask3
        category = category+flag*mask
        flag= flag+1
    return category

# all_prompt
def show_demo(size,anns):
    if len(anns) == 0:
        return (np.ones((size[1],size[0],3))*255).astype(np.uint8),(np.ones((size[1],size[0]))).astype(np.uint16)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    images = np.zeros((size[1],size[0],3))
    images2 = np.zeros((size[1],size[0],1))

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = (np.random.random((1, 3))*255).astype(np.uint8).tolist()[0]

        for i in range(3):
            img[:,:,i] = color_mask[i]
        images = images+img*m.reshape(size[1],size[0],1)

        # images2 = images2+img2*m.reshape(size[1],size[0],1)
        # index = np.where(images2 == flag)
        # flag = flag+1

    
    # 由于掩码的RGB是一层层叠加的，所以需要遍历图像统计RGB的个数
    unique_elements = unique3D(images)
    # 根据统计到的RGB类别，将原有的RGB转换为灰度值
    images2 = get_category(images,unique_elements)
 
    return images,images2  

# single prompt
def show_demo2(size,anns):
    if len(anns) == 0:
        return (np.ones((size[1],size[0],3))*255).astype(np.uint8),(np.ones((size[1],size[0]))).astype(np.uint16)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    images = np.zeros((size[1],size[0],3), dtype=np.uint8)
    images2 = np.zeros((size[1],size[0],1))
    
    flag=1
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_color = (np.random.random((1, 3))*255).astype(np.uint8).tolist()[0]

        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        images[m]  = mask_color
        images2[m] = flag
        images2 = np.squeeze(images2)
        flag = flag+1
 
    return images,images2       

