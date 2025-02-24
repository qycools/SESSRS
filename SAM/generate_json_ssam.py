import os
import torch
import argparse
import json
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from tasks import get_sam_json
from PIL import Image
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('SemanticSAM', add_help=False)
    parser.add_argument('--ckpt', default="ckpt/swinl_only_sam_many2many.pth", metavar="FILE", help='path to ckpt', )
    parser.add_argument('-i','--image_dir', default="../GeoSeg/data/Urban/test/images" )
    parser.add_argument('-j','--sam_json_dir', default="../GeoSeg/data/Urban/sam_label_info/ssam" )
    args = parser.parse_args()
    return args

@torch.no_grad()
def get_sam_info(image,level=[0],box_nms = 0.7,min_mask_region_area=100, pred_iou_thresh=0.88,stability_score_thresh=0.92,*args, **kwargs):
    if level == 'All Prompt':
        level = [ 3, 4, 5, 6] 
    else:
        level = [level.split(' ')[-1]]
    
    text_size = image.size[0] if image.size[0] > image.size[1] else image.size[1]
    with torch.autocast(device_type='cuda', dtype=torch.float16): # type: ignore
        sam_info= get_sam_json(model_sam, image,level,text_size,box_nms=box_nms,min_mask_region_area=min_mask_region_area,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh, *args, **kwargs)
        return sam_info

def process_data_json(img_dir,img_name):

    image = Image.open(os.path.join(img_dir,img_name))
    sam_infos = []
    # ssam 没有使用level1,level2
    for level in [3,4,5,6]:

        sam_info = get_sam_info(image,level = f'{level}',box_nms=box_nms,min_mask_region_area =min_region,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh)
        sam_infos.extend(sam_info)
    
    json_data = json.dumps(sam_infos, indent=4, ensure_ascii=False)  

    with open(os.path.join(args.sam_json_dir,img_name.split('.')[0]+'.json'),'w') as f:  
        f.write(json_data)

if __name__ == '__main__':

    min_region = 500
    box_nms = 0.6
    pred_iou_thresh = 0.95
    stability_score_thresh = 0.95

    args = parse_option()
    cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
        'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

    sam_cfg=cfgs['L']
    opt = load_opt_from_config_file(sam_cfg)
    model_sam = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()

    os.makedirs(args.sam_json_dir,exist_ok=True)
    val_lines = os.listdir(args.image_dir)

    for val_line in tqdm(val_lines):
        process_data_json(args.image_dir,val_line)
    