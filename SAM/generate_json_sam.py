import os
import torch
import argparse
from segment_anything import sam_model_registry,get_sam_json
import json
from PIL import Image
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('SAM Demo', add_help=False)
    parser.add_argument('-i','--image_dir', default="../GeoSeg/data/Urban/test/images" )
    parser.add_argument('-j','--sam_json_dir', default="../GeoSeg/data/Urban/sam_label_info/sam" )
    args = parser.parse_args()
    return args

@torch.no_grad()
def get_sam_info(image,box_nms = 0.7,min_mask_region_area=100, pred_iou_thresh=0.88,stability_score_thresh=0.92,*args, **kwargs):
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        sam_info= get_sam_json(sam, image,box_nms=box_nms,min_mask_region_area=min_mask_region_area,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh, *args, **kwargs)
        return sam_info

def process_data_json(image_file,img_name):

    img = Image.open(os.path.join(image_file,img_name))
    sam_info = get_sam_info(img,box_nms=box_nms,min_mask_region_area =min_region,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh)
    json_data = json.dumps(sam_info, indent=4, ensure_ascii=False)  

    with open(os.path.join(args.sam_json_dir,img_name.split('.')[0]+'.json'),'w') as f:  
        f.write(json_data)


if __name__ == '__main__':

    min_region = 500
    box_nms = 0.6
    pred_iou_thresh = 0.95
    stability_score_thresh = 0.95

    args = parse_option()
    sam_checkpoint = {'vit_h':"ckpt/sam_vit_h_4b8939.pth"}
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint[model_type])
    sam.to(device=device)

    os.makedirs(args.sam_json_dir,exist_ok=True)
    val_lines = os.listdir(args.image_dir)

    for val_line in tqdm(val_lines):
        process_data_json(args.image_dir,val_line)
    