import os
import torch
import argparse
from segment_anything import sam_model_registry,get_sam_label
from PIL import Image
from tqdm import tqdm
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser('SAM Demo', add_help=False)
    parser.add_argument('-i','--image_dir', default="../GeoSeg/data/Urban/test/images" )
    parser.add_argument('-f','--sam_img_dir', default="../GeoSeg/data/Urban/sam_label/samhq" )
    args = parser.parse_args()
    return args

@torch.no_grad()
def get_sam_info(image,box_nms = 0.7,min_mask_region_area=100, pred_iou_thresh=0.88,stability_score_thresh=0.92,*args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        label,blend,label_P= get_sam_label(sam,image,box_nms=box_nms,min_mask_region_area=min_mask_region_area,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh, *args, **kwargs)
        return label,blend,label_P

def process_data_img(image_file,img_name):
    img_name = img_name[:-4]+'.png'
    
    img = Image.open(os.path.join(image_file,img_name))
    label,blend,label_P = get_sam_info(img,box_nms=box_nms,min_mask_region_area =min_region,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh)
    label.save(os.path.join(args.lbl_dir,img_name))
    np.save(os.path.join(args.lblp_dir,img_name.replace('.png','.npy')),label_P)
    blend.save(os.path.join(args.bld_dir,img_name))


def make_dirs():
    os.makedirs(args.sam_img_dir,exist_ok=True)
    os.makedirs(args.lbl_dir,exist_ok=True)
    os.makedirs(args.bld_dir,exist_ok=True)
    os.makedirs(args.lblp_dir,exist_ok=True)

if __name__ == '__main__':

    min_region = 500
    box_nms = 0.6
    pred_iou_thresh = 0.85
    stability_score_thresh = 0.85

    args = parse_option()
    sam_checkpoint = {'vit_h':"ckpt/sam_hq_vit_h.pth"}
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint[model_type])
    sam.to(device=device)

    args.lbl_dir = os.path.join(args.sam_img_dir,'label')
    args.bld_dir = os.path.join(args.sam_img_dir,'blend')
    args.lblp_dir = os.path.join(args.sam_img_dir,'label_p')
    make_dirs()
    val_lines = os.listdir(args.image_dir)

    for val_line in tqdm(val_lines):
        process_data_img(args.image_dir,val_line)
    