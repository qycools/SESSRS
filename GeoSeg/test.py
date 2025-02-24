import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import ttach as tta
import multiprocessing.pool as mpp
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import multiprocessing
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

def img_writer(inp):
    (mask,output_path, mask_name,rgb) = inp
    if rgb:

        mask_RGB_name_png = os.path.join(output_path ,'pre_rgb',mask_name+ '.png')
        mask_rgb_png = config.label2rgb(mask)
        mask_rgb_png = cv2.cvtColor(mask_rgb_png, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_RGB_name_png, mask_rgb_png)
    else:
        mask_P_name_png = os.path.join(output_path ,'pre_p',mask_name+ '.png')
        mask_p_png = mask.astype(np.uint8)
        # mask_p_png = cv2.cvtColor(mask_p_png, cv2.COLOR_GRAY2BGR)
        mask_p_png = Image.fromarray(mask_p_png,'P')
        mask_p_png.putpalette(config.palette)
        mask_p_png.save(mask_P_name_png)
        
# 测试集更改3，一些参数
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", default='config/Urban/fanet.py', type=Path, help="Path to  config")
    # arg("-o", "--output_path", default='fig_results/Urban/fanet', type=Path, help="Path where to save resulting masks.")
    arg("-t", "--tta", help="Test time augmentation.", default="d4", choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("-g", "--gpu_index", default = 0,type=int)    
    arg("--rgb", default= False)
    # arg("--rgb",  help="whether output rgb masks", action='store_true')
    return parser.parse_args()

def makedirs():
    os.makedirs(args.output_path,exist_ok=True)
    if args.rgb:
        os.makedirs(os.path.join(args.output_path, 'pre_rgb'),exist_ok=True)
    else:
        os.makedirs(os.path.join(args.output_path, 'pre_p'),exist_ok=True)

def multipool(function, args,processes=40):
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


if __name__ == "__main__":

    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path = 'fig_results/'+ args.config_path.split('/')[-2]+ args.config_path.split('/')[-1].split('.')[0]
    args.output_path.mkdir(exist_ok=True, parents=True)
    gpu = args.gpu_index
    device = f'cuda:{gpu}'
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    # model.cuda()
    model.to(device)
    # model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'),map_location= torch.device('cuda:0'),config=config)
    # model.cuda()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    makedirs()
   
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].to(device))
            image_ids = input['img_id']
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path),mask_name,args.rgb))
                # img_writer(results[0])

    multipool(img_writer, results)

