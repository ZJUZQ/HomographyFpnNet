#coding=utf-8
"""
"""
import os
import time
import sys
import glob
import _init_path
import argparse
import tensorflow as tf
import pdb
import cv2
import random
import numpy as np

from homographynet.model import create_homographynet_vgg, create_homographynet_mobilenet_v2, create_homographynet_vgg_fpn

MODEL_MAP = {'vgg': create_homographynet_vgg,
            'vgg_fpn': create_homographynet_vgg_fpn,
            'mobilenet_v2': create_homographynet_mobilenet_v2,
            }

target_size = (128, 128)

def generate_points(image_size):
    """
    image_size: (w, h)
    """
    w, h = target_size
    bolder_h = h // 4
    bolder_w = w // 4
    #print(image_size)
    x1 = random.randint(bolder_w, image_size[0] - w - bolder_w)
    y1 = random.randint(bolder_h, image_size[1] - h - bolder_h)
    x2 = x1 + w - 1
    y2 = y1 + h - 1

    pts = []
    # in order of [pt_tl, pt_tr, pt_bl, pt_br]
    for x, y in [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]:
        perturb_x = random.randint(-bolder_w, bolder_w)
        perturb_y = random.randint(-bolder_h, bolder_h)
        pts.append((x+perturb_x, y+perturb_y))
    
    return np.array(pts)

def warp(img, pts_in_orig, pts_in_warp, target_size):
    """
    """
    M = cv2.getPerspectiveTransform(np.float32(pts_in_orig), np.float32(pts_in_warp))
    return cv2.warpPerspective(img, M, target_size, flags=cv2.INTER_CUBIC)

def loader(data_dir, num_samples_per_image=1):
    for image_path in glob.iglob(os.path.join(data_dir, '*.jpg')):
        img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_orig.shape[0] < target_size[1]*3//2 or img_orig.shape[1] < target_size[0]*3//2:
            continue

        # in order of [pt_tl, pt_tr, pt_bl, pt_br]
        w, h = target_size
        pts_in_warp = np.array([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])

        for _ in range(num_samples_per_image):

            pts_in_orig_nonrs = generate_points(img_orig.shape[:2][::-1])
            img_warp = warp(img_orig, pts_in_orig_nonrs, pts_in_warp, target_size)

            yield img_warp, pts_in_warp, img_orig, pts_in_orig_nonrs


def predict_pts_in_orig(img_orig, img_patch_b, model_dict, img_orig_show=None):
    """
    return 4 predicted points in img_orig
    """
    image_pairs_for_predict = []
    pts_pred_nonrs_list = []

    # in order of [pt_tl, pt_tr, pt_bl, pt_br]
    w, h = target_size
    pts_in_warp = np.array([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])

    ######
    #  Model_1: input of (128, 128, 2) (resized original image + original warp image)
    #           outputs: (8,) normalized 4 pts in resized original image
    #####
    img_patch_a = cv2.resize(img_orig, target_size, interpolation=cv2.INTER_CUBIC)
    img_pair = np.stack((img_patch_a, img_patch_b), -1)
    image_pairs_for_predict.append(img_pair)
    img_pair_norm = (img_pair - 127.5) / 127.5
    pts_pred_1 = model_dict['model_1'].predict(img_pair_norm[np.newaxis])
    pts_pred_1_nonrs = pts_pred_1.reshape(-1, 2) * np.array(img_orig.shape[:2][::-1]).reshape(-1, 2)
    pts_pred_nonrs_list.append(pts_pred_1_nonrs)

    if 'model_2' in model_dict:
        M_1 = cv2.getPerspectiveTransform(np.float32(pts_pred_1_nonrs), np.float32(pts_in_warp))
        M_1_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_1_nonrs))
        img_patch_1_a = cv2.warpPerspective(img_orig, M_1, target_size, flags=cv2.INTER_CUBIC)
        img_pair_1 = np.stack((img_patch_1_a, img_patch_b), -1)

        image_pairs_for_predict.append(img_pair_1)
        img_pair_norm = (img_pair_1 - 127.5) / 127.5
        pts_pred_2 = model_dict['model_2'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_2 = pts_pred_2.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_2_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_2).reshape(-1,1,2), M_1_inv)
        pts_pred_2_nonrs = pts_pred_2_nonrs.reshape(-1, 2)
        pts_pred_nonrs_list.append(pts_pred_2_nonrs)
    
    if 'model_3' in model_dict:
        M_2 = cv2.getPerspectiveTransform(np.float32(pts_pred_2_nonrs), np.float32(pts_in_warp))
        M_2_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_2_nonrs))
        img_patch_2_a = cv2.warpPerspective(img_orig, M_2, target_size, flags=cv2.INTER_CUBIC)
        img_pair_2 = np.stack((img_patch_2_a, img_patch_b), -1)

        image_pairs_for_predict.append(img_pair_2)
        img_pair_norm = (img_pair_2 - 127.5) / 127.5
        pts_pred_3 = model_dict['model_3'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_3 = pts_pred_3.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_3_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_3).reshape(-1,1,2), M_2_inv)
        pts_pred_3_nonrs = pts_pred_3_nonrs.reshape(-1, 2)
        pts_pred_nonrs_list.append(pts_pred_3_nonrs)
    
    if 'model_4' in model_dict:
        M_3 = cv2.getPerspectiveTransform(np.float32(pts_pred_3_nonrs), np.float32(pts_in_warp))
        M_3_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_3_nonrs))
        img_patch_3_a = cv2.warpPerspective(img_orig, M_3, target_size, flags=cv2.INTER_CUBIC)
        img_pair_3 = np.stack((img_patch_3_a, img_patch_b), -1)

        image_pairs_for_predict.append(img_pair_3)
        img_pair_norm = (img_pair_3 - 127.5) / 127.5
        pts_pred_4 = model_dict['model_4'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_4 = pts_pred_4.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_4_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_4).reshape(-1,1,2), M_3_inv)
        pts_pred_4_nonrs = pts_pred_4_nonrs.reshape(-1, 2)
        pts_pred_nonrs_list.append(pts_pred_4_nonrs)
    
    if 'model_5' in model_dict:
        M_4 = cv2.getPerspectiveTransform(np.float32(pts_pred_4_nonrs), np.float32(pts_in_warp))
        M_4_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_4_nonrs))
        img_patch_4_a = cv2.warpPerspective(img_orig, M_4, target_size, flags=cv2.INTER_CUBIC)
        img_pair_4 = np.stack((img_patch_4_a, img_patch_b), -1)

    return image_pairs_for_predict, pts_pred_nonrs_list


def main(args):
    input_shape = (target_size[1], target_size[0], 2) # e.g., (128, 128, 2)

    model_dict = {}
    model_1 = MODEL_MAP[args.model_name](weights_file=args.model_stage_1, input_shape=input_shape, 
            pooling=args.pooling, alpha=args.alpha)
    model_1.summary()
    model_dict['model_1'] = model_1
    if args.num_models >= 2:
        model_2 = MODEL_MAP[args.model_name](weights_file=args.model_stage_2, input_shape=input_shape, 
                pooling=args.pooling, alpha=args.alpha)
        model_2.summary()
        model_dict['model_2'] = model_2
    if args.num_models >= 3:
        model_3 = MODEL_MAP[args.model_name](weights_file=args.model_stage_3, input_shape=input_shape, 
                pooling=args.pooling, alpha=args.alpha)
        model_3.summary()
        model_dict['model_3'] = model_3
    if args.num_models >= 4:
        model_4 = MODEL_MAP[args.model_name](weights_file=args.model_stage_4, input_shape=input_shape, 
                pooling=args.pooling, alpha=args.alpha)
        model_4.summary()
        model_dict['model_4'] = model_4
    
    data_loader = loader(args.img_dir)
    img_warp, _, img_orig, _ = next(data_loader)
    del data_loader

    time_start = time.time()
    NUM_SAMPLES = 5000
    for count in range(NUM_SAMPLES):
        print(count)
        _, _ = predict_pts_in_orig(img_orig, img_warp, model_dict)
    time_end = time.time()

    print('mean time {:.2f} ms'.format((time_end - time_start) / float(NUM_SAMPLES) * 1000))
    
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_models', type=int, help='number of modules')
    parser.add_argument('--model_stage_1', type=str)
    parser.add_argument('--model_stage_2', type=str)
    parser.add_argument('--model_stage_3', type=str)
    parser.add_argument('--model_stage_4', type=str)
    parser.add_argument('--model_name', type=str, default='mobilenet_v2')

    parser.add_argument('img_dir', type=str)

    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()
    main(args)