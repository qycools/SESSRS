import os
import torch
import argparse

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from tasks import get_sam_label
from PIL import Image
from tqdm import tqdm
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser('SemanticSAM', add_help=False)
    parser.add_argument('--ckpt', default="ckpt/swinl_only_sam_many2many.pth", metavar="FILE", help='path to ckpt', )
    parser.add_argument('-i','--image_dir', default="../GeoSeg/data/Urban/test/images" )
    parser.add_argument('-f','--sam_img_dir', default="../GeoSeg/data/Urban/sam_label/ssam" )
    args = parser.parse_args()
    return args


@torch.no_grad()
def get_sam_info(image,level=[0],box_nms = 0.7,min_mask_region_area=100, pred_iou_thresh=0.88,stability_score_thresh=0.92,*args, **kwargs):
    if level == 'All prompt':
        level = [1, 2, 3, 4, 5, 6]
    else:
        level = [level.split(' ')[-1]]
    
    text_size = image.size[0] if image.size[0] > image.size[1] else image.size[1]
    with torch.autocast(device_type='cuda', dtype=torch.float16): # type: ignore

        label,blend,label_P= get_sam_label(model_sam, image,level,text_size,box_nms=box_nms,min_mask_region_area=min_mask_region_area,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh, *args, **kwargs)
        return label,blend,label_P

def process_data_img(image_file,img_name):
    img_name = img_name[:-4]+'.png'
    image = Image.open(os.path.join(image_file,img_name))
    if prompts == 'All prompt':
        label,blend,label_P = get_sam_info(image,level = prompts,box_nms=box_nms,min_mask_region_area =min_region,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh)
        # label.save(os.path.join(args.lbl_dir,'All_prompt',img_name))
        # np.save(os.path.join(args.lblp_dir,'All_prompt',img_name.replace('.png','.npy')),label_P)
        blend.save(os.path.join(args.bld_dir,'All_prompt',img_name))
    else:
        for prompt in prompts:
            label,blend,label_P = get_sam_info(image,level = f'{prompt}',box_nms=box_nms,min_mask_region_area =min_region,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh)
            blend.save(os.path.join(args.bld_dir,f'prompt{prompt}',img_name))



def make_dirs():
    os.makedirs(args.sam_img_dir,exist_ok=True)
    os.makedirs(args.lbl_dir,exist_ok=True)
    os.makedirs(args.bld_dir,exist_ok=True)
    os.makedirs(args.lblp_dir,exist_ok=True)

    for f_dir in [args.lbl_dir,args.bld_dir,args.lblp_dir]:
        for prompt in [1,2,3,4,5,6]:
            os.makedirs(os.path.join(f_dir,f'prompt{prompt}'),exist_ok=True)
        os.makedirs(os.path.join(f_dir,'All_prompt'),exist_ok=True)

if __name__ == '__main__':

    min_region = 500
    box_nms = 0.6
    pred_iou_thresh = 0.95
    stability_score_thresh = 0.95
    prompts = [3,4,5,6]
    # prompts = 'All prompt'
    
    args = parse_option()
    cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
        'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

    sam_cfg=cfgs['L']
    opt = load_opt_from_config_file(sam_cfg)
    model_sam = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()

    args.lbl_dir = os.path.join(args.sam_img_dir,'label')
    args.bld_dir = os.path.join(args.sam_img_dir,'blend')
    args.lblp_dir = os.path.join(args.sam_img_dir,'label_p')

    make_dirs()
    val_lines = os.listdir(args.image_dir)
    
    for val_line in tqdm(val_lines):
        process_data_img(args.image_dir,val_line)


