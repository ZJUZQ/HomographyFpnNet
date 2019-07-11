#coding=utf-8
"""
python train_val/demo_homographynet_reg.py 4 \
    --model_stage_1 pretrained_model/homography_mbv2_crop_128x128_stage_1_avg_alpha_1.0_regression_lr_0.05_bs_64_110k.h5 \
    --model_stage_2 pretrained_model/homography_mbv2_crop_128x128_stage_2_avg_alpha_1.0_regression_lr_0.05_bs_64_70k.h5 \
    --model_stage_3 pretrained_model/homography_mbv2_crop_128x128_stage_3_avg_alpha_1.0_regression_lr_0.05_bs_64_90k.h5 \
    --model_stage_4 pretrained_model/homography_mbv2_crop_128x128_stage_4_avg_alpha_1.0_regression_lr_0.05_bs_64_90k.h5 \
    --img_dir_anns ~/paper_deep_homographynet_crobot/test_set/wall_3_2/
"""
import os
import sys
import glob
import _init_path
import argparse
import tensorflow as tf
import pdb
import cv2
import random
import numpy as np

from homographynet.model import create_homographynet_vgg, create_homographynet_mobilenet_v2 
from homographynet.model import create_homographynet_vgg_fpn

MODEL_MAP = {'vgg': create_homographynet_vgg,
            'vgg_fpn': create_homographynet_vgg_fpn,
            'mobilenet_v2': create_homographynet_mobilenet_v2,
            }

target_size = (128, 128)

parser = argparse.ArgumentParser()
parser.add_argument('num_models', type=int, help='number of modules')
parser.add_argument('--model_stage_1', type=str)
parser.add_argument('--model_stage_2', type=str)
parser.add_argument('--norm_value_stage_2', type=int)
parser.add_argument('--model_stage_3', type=str)
parser.add_argument('--norm_value_stage_3', type=int)
parser.add_argument('--model_stage_4', type=str)
parser.add_argument('--norm_value_stage_4', type=int)

parser.add_argument('--model_name', type=str, default='mobilenet_v2')

parser.add_argument('--img_dir_anns', type=str)
parser.add_argument('--img_warp_file', type=str)

parser.add_argument('--img_dir', type=str)
parser.add_argument('--num_images', type=int, default=10000)
parser.add_argument('--num_samples_per_image', type=int, default=1)

parser.add_argument('--pooling', type=str, default='avg')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--show', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

pts_norm_value_stage_2 = (args.norm_value_stage_2, args.norm_value_stage_2) #(64, 64)
pts_norm_value_stage_3 = (args.norm_value_stage_3, args.norm_value_stage_3) #(32, 32)
pts_norm_value_stage_4 = (args.norm_value_stage_4, args.norm_value_stage_4) #(16, 16)


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

def loader(data_dir, num_samples_per_image):
    num_image_count = 0
    for image_path in glob.iglob(os.path.join(data_dir, '*.jpg')):
        if num_image_count >= args.num_images:
            return
        img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_orig.shape[0] < target_size[1]*3//2 or img_orig.shape[1] < target_size[0]*3//2:
            continue
        num_image_count +=1

        # in order of [pt_tl, pt_tr, pt_bl, pt_br]
        w, h = target_size
        pts_in_warp = np.array([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])

        for _ in range(num_samples_per_image):
            #pts_in_orig_nonrs = np.array([[60, 70], [234, 70], [70, 300], [221, 310]])
            #print('use fix points for tmp')
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

    #for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
    #    img_orig_show = cv2.line(img_orig_show, (int(pts_pred_1_nonrs[k1][0]), int(pts_pred_1_nonrs[k1][1])),
    #                (int(pts_pred_1_nonrs[k2][0]), int(pts_pred_1_nonrs[k2][1])), (0, 0, 255), 2)

    M_1 = cv2.getPerspectiveTransform(np.float32(pts_pred_1_nonrs), np.float32(pts_in_warp))
    M_1_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_1_nonrs))
    img_patch_1_a = cv2.warpPerspective(img_orig, M_1, target_size, flags=cv2.INTER_CUBIC)
    img_pair_1 = np.stack((img_patch_1_a, img_patch_b), -1)

    ######
    #  Model_2: input of (128, 128, 2) (part of original image + original warp image)
    #           outputs: (8,) normalized 4 pts in part of original image, relative to "pts_refer_1"
    #####
    if 'model_2' in model_dict:
        image_pairs_for_predict.append(img_pair_1)
        img_pair_norm = (img_pair_1 - 127.5) / 127.5
        pts_pred_2 = model_dict['model_2'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_2 = pts_pred_2.reshape(-1, 2) * np.array(pts_norm_value_stage_2).reshape(-1, 2) + pts_in_warp

        # pts in original image
        pts_pred_2_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_2).reshape(-1,1,2), M_1_inv)
        pts_pred_2_nonrs = pts_pred_2_nonrs.reshape(-1, 2)
        pts_pred_nonrs_list.append(pts_pred_2_nonrs)

        #for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
        #    img_orig_show = cv2.line(img_orig_show, (int(pts_pred_2_nonrs[k1][0]), int(pts_pred_2_nonrs[k1][1])),
        #                (int(pts_pred_2_nonrs[k2][0]), int(pts_pred_2_nonrs[k2][1])), (0, 100, 255), 2)
        
        M_2 = cv2.getPerspectiveTransform(np.float32(pts_pred_2_nonrs), np.float32(pts_in_warp))
        M_2_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_2_nonrs))
        img_patch_2_a = cv2.warpPerspective(img_orig, M_2, target_size, flags=cv2.INTER_CUBIC)
        img_pair_2 = np.stack((img_patch_2_a, img_patch_b), -1)
    
    if 'model_3' in model_dict:
        image_pairs_for_predict.append(img_pair_2)
        img_pair_norm = (img_pair_2 - 127.5) / 127.5
        pts_pred_3 = model_dict['model_3'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_3 = pts_pred_3.reshape(-1, 2) * np.array(pts_norm_value_stage_3).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_3_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_3).reshape(-1,1,2), M_2_inv)
        pts_pred_3_nonrs = pts_pred_3_nonrs.reshape(-1, 2)
        pts_pred_nonrs_list.append(pts_pred_3_nonrs)

        #for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
        #    img_orig_show = cv2.line(img_orig_show, (int(pts_pred_3_nonrs[k1][0]), int(pts_pred_3_nonrs[k1][1])),
        #                (int(pts_pred_3_nonrs[k2][0]), int(pts_pred_3_nonrs[k2][1])), (0, 200, 255), 2)
        
        M_3 = cv2.getPerspectiveTransform(np.float32(pts_pred_3_nonrs), np.float32(pts_in_warp))
        M_3_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_3_nonrs))
        img_patch_3_a = cv2.warpPerspective(img_orig, M_3, target_size, flags=cv2.INTER_CUBIC)
        img_pair_3 = np.stack((img_patch_3_a, img_patch_b), -1)
    
    if 'model_4' in model_dict:
        image_pairs_for_predict.append(img_pair_3)
        img_pair_norm = (img_pair_3 - 127.5) / 127.5
        pts_pred_4 = model_dict['model_4'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_4 = pts_pred_4.reshape(-1, 2) * np.array(pts_norm_value_stage_4).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_4_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_4).reshape(-1,1,2), M_3_inv)
        pts_pred_4_nonrs = pts_pred_4_nonrs.reshape(-1, 2)
        pts_pred_nonrs_list.append(pts_pred_4_nonrs)

        #for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
        #    img_orig_show = cv2.line(img_orig_show, (int(pts_pred_4_nonrs[k1][0]), int(pts_pred_4_nonrs[k1][1])),
        #                (int(pts_pred_4_nonrs[k2][0]), int(pts_pred_4_nonrs[k2][1])), (255, 0, 255), 2)
        
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

    pts_err_dict = {}
    for k in range(1, len(model_dict)+1):
        pts_err_dict['stage_{}'.format(k)] = []
    count = 0

    if args.show == True:
        cv2.namedWindow('show', 0)
    
    if args.img_dir is not None:
        data_loader = loader(args.img_dir, args.num_samples_per_image)
        for img_warp, pts_in_warp, img_orig, pts_in_orig_nonrs in data_loader:
            count += 1
            print(count)

            image_pairs_for_predict, pts_pred_nonrs_list = predict_pts_in_orig(img_orig, img_warp, model_dict)
            
            for k in range(1, len(pts_pred_nonrs_list)+1):
                pts_err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_nonrs_list[k-1]), -1)))
                pts_err_dict['stage_{}'.format(k)].append(pts_err)
                print(pts_err)

            if args.show == True:
                img_show = np.ones((img_orig.shape[0], img_orig.shape[1]*len(pts_pred_nonrs_list), 3), np.uint8)*255
                for k in range(len(pts_pred_nonrs_list)):
                    img_orig_show = np.copy(img_orig)
                    img_orig_show = cv2.cvtColor(img_orig_show, cv2.COLOR_GRAY2BGR)
                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_orig_show = cv2.line(img_orig_show, (int(pts_in_orig_nonrs[k1][0]), int(pts_in_orig_nonrs[k1][1])),
                                    (int(pts_in_orig_nonrs[k2][0]), int(pts_in_orig_nonrs[k2][1])), (0, 255, 0), 2)
                    
                    #assert os.path.isdir('/Users/jianchong/output/tmp/')             
                    #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', 'img_{}_gt.png'.format(count)), img_orig_show)
                    #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', 'img_{}_warp.png'.format(count)), img_warp)

                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_orig_show = cv2.line(img_orig_show, (int(pts_pred_nonrs_list[k][k1][0]), int(pts_pred_nonrs_list[k][k1][1])),
                                    (int(pts_pred_nonrs_list[k][k2][0]), int(pts_pred_nonrs_list[k][k2][1])), (0, 0, 255), 2)
                    
                    err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_nonrs_list[k]), -1)))
                    cv2.putText(img_orig_show, 'mean corner error: {:.1f}'.format(err), (10, 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,0,255), 1)
                    img_show[:, k*img_orig.shape[1]:(k+1)*img_orig.shape[1], :] = img_orig_show

                    if k == len(pts_pred_nonrs_list) - 1:
                        img_save = np.zeros((img_orig_show.shape[0], img_orig_show.shape[1]+img_warp.shape[1], 3))
                        img_save[:img_warp.shape[0], :img_warp.shape[1], :] = cv2.cvtColor(img_warp, cv2.COLOR_GRAY2BGR)
                        img_save[:, -img_orig_show.shape[1]:, :] = img_orig_show
                        cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', 'img_{}_homographyRegNet.png'.format(count)), img_save)


                
                #### tmp save images ####
                for k in range(len(image_pairs_for_predict)):
                    if k == len(image_pairs_for_predict) - 1:
                        assert os.path.isdir('/Users/jianchong/output/tmp/')
                        img_pair_save = np.ones((target_size[1], target_size[0]*2+10, 3), dtype=np.uint8) * 255
                        img_a = cv2.cvtColor(image_pairs_for_predict[k][:, :, 0], cv2.COLOR_GRAY2BGR)
                        img_b = cv2.cvtColor(image_pairs_for_predict[k][:, :, 1], cv2.COLOR_GRAY2BGR)
                        img_pair_save[:, :target_size[0], :] = img_a
                        img_pair_save[:, -target_size[0]:, :] = img_b
                        #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', 'img_{}_pair_{}.png'.format(count, k)), img_pair_save)
                        #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', 'img_{}_predict_{}.png'.format(count, k)), img_show[:, k*img_orig.shape[1]:(k+1)*img_orig.shape[1], :])
                
                
                #cv2.imshow('show', img_show)
                #key = cv2.waitKey(0)
                #if key in [ord('q'), ord('Q')]:
                #    return
            
    
    elif args.img_dir_anns is not None:
        # for crbot wall evaluate
        ann_file = os.path.join(args.img_dir_anns, 'annotation.txt')
        assert os.path.isfile(ann_file)
        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                if line.strip() == '' or line.strip()[0] == '#':
                    continue
                anns = line.strip().split(' ')
                ann_list.append(anns)

        for anns in ann_list:
            count += 1
            print(count)

            image_name = os.path.join(args.img_dir_anns, anns[0])
            img_orig = cv2.imread(image_name, 0) # read gray image
            # pts order: [p_tl, p_tr, p_bl, p_br]
            pts_in_orig_nonrs = [float(t) for t in anns[1:]]
            pts_in_orig_nonrs = np.array(pts_in_orig_nonrs).reshape(-1, 2)

            w, h = target_size
            pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)

            if args.img_warp_file is not None:
                img_warp_orig = cv2.imread(args.img_warp_file, 0)
                img_warp = cv2.resize(img_warp_orig, target_size, cv2.INTER_CUBIC)

                img_warp_show = cv2.cvtColor(img_warp_orig, cv2.COLOR_GRAY2BGR)
            else:
                M = cv2.getPerspectiveTransform(np.float32(pts_in_orig_nonrs), np.float32(pts_in_warp))
                M_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_in_orig_nonrs))
                img_warp = cv2.warpPerspective(img_orig, M, target_size, flags=cv2.INTER_CUBIC)

                w = int(np.max(pts_in_orig_nonrs[:, 0]) - np.min(pts_in_orig_nonrs[:, 0]) + 1)
                h = int(np.max(pts_in_orig_nonrs[:, 1]) - np.min(pts_in_orig_nonrs[:, 1]) + 1)
                pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)
                img_warp_show = warp(img_orig, pts_in_orig_nonrs, pts_in_warp, (w, h))
                img_warp_show = cv2.cvtColor(img_warp_show, cv2.COLOR_GRAY2BGR)

            image_pairs_for_predict, pts_pred_nonrs_list = predict_pts_in_orig(img_orig, img_warp, model_dict)

            for k in range(1, len(pts_pred_nonrs_list)+1):
                pts_err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_nonrs_list[k-1]), -1)))
                pts_err_dict['stage_{}'.format(k)].append(pts_err)
                print(pts_err)

            if args.show == True:
                img_show = np.ones((img_orig.shape[0], img_orig.shape[1]*len(pts_pred_nonrs_list), 3), np.uint8)*255
                for k in range(len(pts_pred_nonrs_list)):
                    img_orig_show = np.copy(img_orig)
                    img_orig_show = cv2.cvtColor(img_orig_show, cv2.COLOR_GRAY2BGR)
                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_orig_show = cv2.line(img_orig_show, (int(pts_in_orig_nonrs[k1][0]), int(pts_in_orig_nonrs[k1][1])),
                                    (int(pts_in_orig_nonrs[k2][0]), int(pts_in_orig_nonrs[k2][1])), (0, 255, 0), 4)
                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_orig_show = cv2.line(img_orig_show, (int(pts_pred_nonrs_list[k][k1][0]), int(pts_pred_nonrs_list[k][k1][1])),
                                    (int(pts_pred_nonrs_list[k][k2][0]), int(pts_pred_nonrs_list[k][k2][1])), (0, 0, 255), 4)
                    cv2.putText(img_orig_show, 'mean corner error: {:.1f}'.format(pts_err_dict['stage_{}'.format(k+1)][-1]), (10, 60), 
                        cv2.FONT_HERSHEY_COMPLEX, 1.6,(0,0,255), 2)

                    img_show[:, k*img_orig.shape[1]:(k+1)*img_orig.shape[1], :] = img_orig_show
                    if k == len(pts_pred_nonrs_list)-1:
                        img_save = np.zeros((img_orig_show.shape[0], img_orig_show.shape[1]+img_warp_show.shape[1], 3))
                        img_save[:img_warp_show.shape[0], :img_warp_show.shape[1], :] = img_warp_show
                        img_save[:, -img_orig_show.shape[1]:, :] = img_orig_show
                        cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_image_predict_{}.png'.format(count, k)), img_save)

                #### tmp save images ####
                
                for k in range(len(image_pairs_for_predict)):
                    assert os.path.isdir('/Users/jianchong/output/tmp/')
                    img_pair_save = np.ones((target_size[1], target_size[0]*2+10, 3), dtype=np.uint8) * 255
                    img_a = cv2.cvtColor(image_pairs_for_predict[k][:, :, 0], cv2.COLOR_GRAY2BGR)
                    img_b = cv2.cvtColor(image_pairs_for_predict[k][:, :, 1], cv2.COLOR_GRAY2BGR)
                    img_pair_save[:, :target_size[0], :] = img_a
                    img_pair_save[:, -target_size[0]:, :] = img_b
                    if k == 0:
                        pass
                        #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_image_a_{}.png'.format(count, k)), img_a)
                        #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_image_b_{}.png'.format(count, k)), img_b)
                        #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_image_pair_{}.png'.format(count, k)), img_pair_save)
                
                #cv2.imshow('show', img_show)
                #key = cv2.waitKey(0)
                #if key in [ord('q'), ord('Q')]:
                #    return
   
    print('')
    print('*'*80)
    print('Total num samples: {}'.format(count))

    for k in range(1, len(pts_err_dict)+1):
        with open('/Users/jianchong/output/tmp/'+'pts_err_stage_{}.txt'.format(k), 'w') as f:
            for t in pts_err_dict['stage_{}'.format(k)]:
                f.write('{}\n'.format(t))

        print('stage_{}'.format(k))
        print('  mean pts error: {:.2f} [pixel]'.format(np.mean(pts_err_dict['stage_{}'.format(k)])))
        print('  num samples: {}'.format(len(pts_err_dict['stage_{}'.format(k)])))


if __name__ == '__main__':
    if args.debug == True:
        pdb.set_trace()
    main(args)
    if args.show == True:
        cv2.destroyAllWindows()