import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random
from tqdm import tqdm

SEED = 42
PALETTE = [255, 255, 255,0, 0, 255,0, 255, 255,0, 255, 0,255, 204, 0,255,0,0,0, 0, 0]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
Boundary = np.array([0, 0, 0]) # label 6
num_classes = 6


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="/ExtRaid/data_qy/dataset_orig/potsdam/2_Ortho_RGB")
    parser.add_argument("--mask-dir", default="/ExtRaid/data_qy/dataset_orig/potsdam/5_Labels_all")
    parser.add_argument("--read_txt", default="/ExtRaid/data_qy/dataset_orig/potsdam/val.txt")
    parser.add_argument("--output-img-dir", default="data/potsdam_a/test/images")
    parser.add_argument("--output-mask-dir", default="data/potsdam_a/test/masks")
    parser.add_argument("--write_txt", default="data/potsdam_a/val.txt")
    parser.add_argument("--eroded", default = False)
    # parser.add_argument("--gt", action='store_true')  # output RGB mask
    # parser.add_argument("--rgb-image", action='store_true')  # use Potsdam RGB format images
    parser.add_argument("--mode", type=str, default='val')
    parser.add_argument("--val-scale", type=float, default=1.0)  # ignore
    parser.add_argument("--split-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    return parser.parse_args()


def get_img_mask_padded(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad

    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT,
                               value=0)(image=img)
    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT,
                                value=0)(image=mask)

    img_pad, mask_pad = pad_img['image'], pad_mask['image']
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, mask_pad

def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask

def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        h_vlip = RandomHorizontalFlip(p=1.0)
        v_vlip = RandomVerticalFlip(p=1.0)
        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())

        image_list_train = [image, image_h_vlip, image_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], mask_list_train[i], patch_size, mode)
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        # rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        # image, mask = rescale(image.copy()), rescale(mask.copy())
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
        # image, mask = get_img_mask_padded(image.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())

        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def car_aug(image, mask):
    assert image.shape[:2] == mask.shape
    resize_crop_1 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.25), height=int(image.shape[1] * 1.25)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), mask=mask.copy())
    resize_crop_2 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.5), height=int(image.shape[1] * 1.5)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), mask=mask.copy())
    resize_crop_3 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.75), height=int(image.shape[1] * 1.75)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), mask=mask.copy())
    resize_crop_4 = albu.Compose([albu.Resize(width=int(image.shape[0] * 2.0), height=int(image.shape[1] * 2.0)),
                                  albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(), mask=mask.copy())
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
    image_resize_crop_1, mask_resize_crop_1 = resize_crop_1['image'], resize_crop_1['mask']
    image_resize_crop_2, mask_resize_crop_2 = resize_crop_2['image'], resize_crop_2['mask']
    image_resize_crop_3, mask_resize_crop_3 = resize_crop_3['image'], resize_crop_3['mask']
    image_resize_crop_4, mask_resize_crop_4 = resize_crop_4['image'], resize_crop_4['mask']
    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
    image_list = [image, image_resize_crop_1, image_resize_crop_2, image_resize_crop_3,
                  image_resize_crop_4, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_resize_crop_1, mask_resize_crop_2, mask_resize_crop_3,
                 mask_resize_crop_4, mask_vflip, mask_hflip, mask_rotate]

    return image_list, mask_list


def patch_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir,mode, val_scale, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0][:-4]
    mask_filename = img_filename
    # mask_filename = os.path.splitext(os.path.basename(mask_path))[0][:-6]

    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')

    # print(mask)
    # print(img_path)
    # print(img.size, mask.size)
    # img and mask shape: WxHxC
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
                                          val_scale=val_scale, mode=mode)
   
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                x_start,y_start,x_end,y_end = x,y,x + split_size,y + split_size
                if y_end>img.shape[0]:
                    y_end = img.shape[0]
                    y_start = y_end - split_size
                if x_end>img.shape[1]:
                    x_end = img.shape[1]
                    x_start = x_end - split_size
                img_tile_cut = img[y_start:y_end , x_start:x_end]
                mask_tile_cut = mask[y_start:y_end , x_start:x_end]
                img_tile, mask_tile = img_tile_cut, mask_tile_cut

                out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(img_filename, m, k))
                cv2.imwrite(out_img_path, img_tile)

                out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                mask_tile = Image.fromarray(mask_tile)
                mask_tile.putpalette(PALETTE)
                mask_tile.save(out_mask_path)
                    # cv2.imwrite(out_mask_path, mask_tile)
                k += 1
                if x_end == img.shape[1]:
                    break
            if y_end == img.shape[0]:
                break
                
def file2txt():
    files = sorted(os.listdir(args.output_img_dir))
    with open(args.write_txt, 'w') as f:
        for file in files:
            f.write(file+'\n')

def multipool(function, pre_p_paths,processes=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool(processes=processes)
    with tqdm(total=len(pre_p_paths)) as pbar:
    # 使用imap_unordered来执行任务并迭代结果
        for result in pool.imap_unordered(function, pre_p_paths):
            # 这里可以处理任务结果
            # 在任务完成时更新进度条
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()

if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    # eroded = args.eroded
    # gt = args.gt
    # rgb_image = args.rgb_image
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    read_txt = args.read_txt

    with open(read_txt,'r') as f:
        mask_paths = f.readlines()

    img_paths = [os.path.join(imgs_dir,mask_path.split('label')[0]+'RGB.tif') for mask_path in mask_paths]

    if args.eroded:
        mask_paths = [os.path.join(masks_dir,mask_path.split('label')[0]+'label_noBoundary.tif') for mask_path in mask_paths]
    else:
        mask_paths = [os.path.join(masks_dir,mask_path.split('label')[0]+'label.tif') for mask_path in mask_paths]
    
    img_paths.sort()
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)


    inps = sorted([(img_path, mask_path, imgs_output_dir, masks_output_dir, mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)])
    # for inp in inps:
    #     patch_format(inp)
    multipool(patch_format,inps,7)
    # file2txt()


