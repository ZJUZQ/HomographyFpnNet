#coding=utf-8
"""
"""
import os
import sys
import glob
import cv2
import numpy as np
import argparse
import pdb

import _init_path
from homographynet.model import create_homographynet_vgg, create_homographynet_mobilenet_v2
from homographynet.model import create_homographynet_vgg_fpn
from dataset.generate_dataset_HomographyRegNet_v2 import generate_points, warp, pack

MODEL_MAP = {'vgg': create_homographynet_vgg,
            'vgg_fpn': create_homographynet_vgg_fpn,
            'mobilenet_v2': create_homographynet_mobilenet_v2,
            }
target_size = (128, 128) # (w, h)

parser = argparse.ArgumentParser()
parser.add_argument('num_models', type=int, help='number of modules')
parser.add_argument('images_dir', type=str)
parser.add_argument('num_samples_per_image', type=int)

parser.add_argument('--model_stage_1', type=str)
parser.add_argument('--model_stage_2', type=str)
parser.add_argument('--model_stage_3', type=str)
parser.add_argument('--model_stage_4', type=str)

parser.add_argument('--gen_next_stage_data', action='store_true', default=False)
parser.add_argument('--output_data_dir', type=str)

parser.add_argument('--model_name', type=str, default='mobilenet_v2')
parser.add_argument('--pooling', type=str, default='avg')
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('--show', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()


def loader(data_dir, num_samples_per_image):
    for image_path in glob.iglob(os.path.join(data_dir, '*.jpg')):
        img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h_, w_ = img_orig.shape[:2]

        if img_orig.shape[0] < target_size[1]*3//2 or img_orig.shape[1] < target_size[0]*3//2:
            continue

        # in order of [pt_tl, pt_tr, pt_bl, pt_br]
        w, h = target_size
        pts_in_warp = np.array([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])

        for _ in range(num_samples_per_image):

            pts_in_orig_nonrs = generate_points(img_orig.shape[:2][::-1])
            img_warp = warp(img_orig, pts_in_orig_nonrs, pts_in_warp, target_size)

            # direct resize img and pts_in_orig to target_size
            img_resize = cv2.resize(img_orig, target_size, interpolation=cv2.INTER_CUBIC)
            pts_in_orig = np.copy(pts_in_orig_nonrs.reshape(-1, 2))
            pts_in_orig[:, 0] = pts_in_orig[:, 0] * float(w) / float(w_)
            pts_in_orig[:, 1] = pts_in_orig[:, 1] * float(h) / float(h_)

            img_pair = np.stack((img_resize, img_warp), -1)
            
            yield img_pair, pts_in_orig, pts_in_warp, img_orig, pts_in_orig_nonrs

def process_image(img_pair, pts_in_orig, pts_in_warp, img_orig, pts_in_orig_nonrs, model_dict):
    """
    img_pair: original image resized to (128, 128) + original warped image (128, 128)
    pts_in_orig: groundtruth points in resized original image (128, 128)
    pts_in_orig_nonrs: groundtruth points in non-resized image
    """
    err_dict = {}
    flag = True # whether useful data

    img_orig_show = np.copy(img_orig)
    img_orig_show = cv2.cvtColor(img_orig_show, cv2.COLOR_GRAY2BGR)
    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
        img_orig_show = cv2.line(img_orig_show, (int(pts_in_orig_nonrs[k1][0]), int(pts_in_orig_nonrs[k1][1])),
                    (int(pts_in_orig_nonrs[k2][0]), int(pts_in_orig_nonrs[k2][1])), (0, 255, 0), 2)
    #img_orig_show = cv2.putText(img_orig_show, 'gt', (int(pts_in_orig_nonrs[0][0]), int(pts_in_orig_nonrs[0][1])),
    #            cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 1)
    ######
    #  Model_1: input of (128, 128, 2) (resized original image + original warp image)
    #           outputs: (8,) normalized 4 pts in resized original image
    #####
    img_pair_norm = (img_pair - 127.5) / 127.5
    pts_pred_1 = model_dict['model_1'].predict(img_pair_norm[np.newaxis])
    pts_pred_1_rs = pts_pred_1.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) # predicted pts in resized (128, 128)
    pts_pred_1_nonrs = pts_pred_1.reshape(-1, 2) * np.array(img_orig.shape[:2][::-1]).reshape(-1, 2)
    # clip
    pts_pred_1_nonrs[:, 0] = np.clip(pts_pred_1_nonrs[:, 0], 0, img_orig.shape[1])
    pts_pred_1_nonrs[:, 1] = np.clip(pts_pred_1_nonrs[:, 1], 0, img_orig.shape[0])

    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
        img_orig_show = cv2.line(img_orig_show, (int(pts_pred_1_nonrs[k1][0]), int(pts_pred_1_nonrs[k1][1])),
                    (int(pts_pred_1_nonrs[k2][0]), int(pts_pred_1_nonrs[k2][1])), (0, 0, 255), 2)
    #img_orig_show = cv2.putText(img_orig_show, 'one', (int(pts_pred_1_nonrs[0][0]), int(pts_pred_1_nonrs[0][1])),
    #            cv2.FONT_HERSHEY_COMPLEX, 4, (0,0,255), 1)

    errs = np.sqrt(np.sum(np.square(pts_in_orig - pts_pred_1_rs), -1))
    err_dict['error_stage_1'] = errs.tolist()
    errs_nonrs = np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_1_nonrs), -1))
    err_dict['error_stage_1_nonrs'] = errs_nonrs.tolist()

    M_1 = cv2.getPerspectiveTransform(np.float32(pts_pred_1_nonrs), np.float32(pts_in_warp))
    M_1_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_1_nonrs))
    img_patch_1_a = cv2.warpPerspective(img_orig, M_1, target_size, flags=cv2.INTER_CUBIC)
    img_pair_1 = np.stack((img_patch_1_a, img_pair[:, :, 1]), -1)

    pts_gt_1 = cv2.perspectiveTransform(np.float32(pts_in_orig_nonrs).reshape(-1,1,2), M_1)
    pts_gt_1 = pts_gt_1.reshape(-1, 2) - pts_in_warp
    pts_gt_1_norm = pts_gt_1 / np.array(target_size).reshape(-1, 2)

    # for generate next stage dataset
    img_pair_new = img_pair_1
    pts_new_norm = pts_gt_1_norm

    if args.num_models >= 2:
        ######
        #  Model_2: input of (128, 128, 2) (part of original image + original warp image)
        #           outputs: (8,) normalized 4 pts in part of original image, relative to "pts_refer_1"
        #####
        img_pair_norm = (img_pair_1 - 127.5) / 127.5
        pts_pred_2 = model_dict['model_2'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_2 = pts_pred_2.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_2_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_2).reshape(-1,1,2), M_1_inv)
        pts_pred_2_nonrs = pts_pred_2_nonrs.reshape(-1, 2)
        # clip
        pts_pred_2_nonrs[:, 0] = np.clip(pts_pred_2_nonrs[:, 0], 0, img_orig.shape[1])
        pts_pred_2_nonrs[:, 1] = np.clip(pts_pred_2_nonrs[:, 1], 0, img_orig.shape[0])

        for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
            img_orig_show = cv2.line(img_orig_show, (int(pts_pred_2_nonrs[k1][0]), int(pts_pred_2_nonrs[k1][1])),
                        (int(pts_pred_2_nonrs[k2][0]), int(pts_pred_2_nonrs[k2][1])), (0, 100, 255), 2)
        #img_orig_show = cv2.putText(img_orig_show, 'two', (int(pts_pred_2_nonrs[0][0]), int(pts_pred_2_nonrs[0][1])),
        #        cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

        errs_nonrs = np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_2_nonrs), -1))
        err_dict['error_stage_2_nonrs'] = errs_nonrs.tolist()

        M_2 = cv2.getPerspectiveTransform(np.float32(pts_pred_2_nonrs), np.float32(pts_in_warp))
        M_2_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_2_nonrs))
        img_patch_2_a = cv2.warpPerspective(img_orig, M_2, target_size, flags=cv2.INTER_CUBIC)
        img_pair_2 = np.stack((img_patch_2_a, img_pair[:, :, 1]), -1)

        pts_gt_2 = cv2.perspectiveTransform(np.float32(pts_in_orig_nonrs).reshape(-1,1,2), M_2)
        pts_gt_2 = pts_gt_2.reshape(-1, 2) - pts_in_warp
        pts_gt_2_norm = pts_gt_2 / np.array(target_size).reshape(-1, 2)

        # for generate next stage dataset
        img_pair_new = img_pair_2
        pts_new_norm = pts_gt_2_norm
        
        # 否则预测的点顺序会错位
        if np.max(np.abs(pts_new_norm[:, 0])) >= 0.5 or np.max(np.abs(pts_new_norm[:, 1])) >= 0.5:
            print('skip')
            flag = False

    if args.num_models >= 3:
        ######
        #  Model_3: input of (128, 128, 2) (part of original image + original warp image)
        #           outputs: (8,) normalized 4 pts in part of original image, relative to "pts_refer_1"
        #####
        img_pair_norm = (img_pair_2 - 127.5) / 127.5
        pts_pred_3 = model_dict['model_3'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_3 = pts_pred_3.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_3_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_3).reshape(-1,1,2), M_2_inv)
        pts_pred_3_nonrs = pts_pred_3_nonrs.reshape(-1, 2)
        # clip
        pts_pred_3_nonrs[:, 0] = np.clip(pts_pred_3_nonrs[:, 0], 0, img_orig.shape[1])
        pts_pred_3_nonrs[:, 1] = np.clip(pts_pred_3_nonrs[:, 1], 0, img_orig.shape[0])

        for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
            img_orig_show = cv2.line(img_orig_show, (int(pts_pred_3_nonrs[k1][0]), int(pts_pred_3_nonrs[k1][1])),
                        (int(pts_pred_3_nonrs[k2][0]), int(pts_pred_3_nonrs[k2][1])), (0, 200, 255), 2)
        #img_orig_show = cv2.putText(img_orig_show, 'two', (int(pts_pred_3_nonrs[0][0]), int(pts_pred_3_nonrs[0][1])),
        #        cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

        errs_nonrs = np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_3_nonrs), -1))
        err_dict['error_stage_3_nonrs'] = errs_nonrs.tolist()

        M_3 = cv2.getPerspectiveTransform(np.float32(pts_pred_3_nonrs), np.float32(pts_in_warp))
        M_3_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_3_nonrs))
        img_patch_3_a = cv2.warpPerspective(img_orig, M_3, target_size, flags=cv2.INTER_CUBIC)
        img_pair_3 = np.stack((img_patch_3_a, img_pair[:, :, 1]), -1)

        pts_gt_3 = cv2.perspectiveTransform(np.float32(pts_in_orig_nonrs).reshape(-1,1,2), M_3)
        pts_gt_3 = pts_gt_3.reshape(-1, 2) - pts_in_warp
        pts_gt_3_norm = pts_gt_3 / np.array(target_size).reshape(-1, 2)

        # for generate next stage dataset
        img_pair_new = img_pair_3
        pts_new_norm = pts_gt_3_norm

        if np.max(np.abs(pts_new_norm[:, 0])) >= 0.5 or np.max(np.abs(pts_new_norm[:, 1])) >= 0.5:
            print('skip')
            flag = False
    
    if args.num_models >= 4:
        ######
        #  Model_4: input of (128, 128, 2) (part of original image + original warp image)
        #           outputs: (8,) normalized 4 pts in part of original image, relative to "pts_refer_1"
        #####
        img_pair_norm = (img_pair_3 - 127.5) / 127.5
        pts_pred_4 = model_dict['model_4'].predict(img_pair_norm[np.newaxis])
        # pts in img_warp
        pts_pred_4 = pts_pred_4.reshape(-1, 2) * np.array(target_size).reshape(-1, 2) + pts_in_warp
        # pts in original image
        pts_pred_4_nonrs = cv2.perspectiveTransform(np.float32(pts_pred_4).reshape(-1,1,2), M_3_inv)
        pts_pred_4_nonrs = pts_pred_4_nonrs.reshape(-1, 2)
        # clip
        pts_pred_4_nonrs[:, 0] = np.clip(pts_pred_4_nonrs[:, 0], 0, img_orig.shape[1])
        pts_pred_4_nonrs[:, 1] = np.clip(pts_pred_4_nonrs[:, 1], 0, img_orig.shape[0])

        for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
            img_orig_show = cv2.line(img_orig_show, (int(pts_pred_4_nonrs[k1][0]), int(pts_pred_4_nonrs[k1][1])),
                        (int(pts_pred_4_nonrs[k2][0]), int(pts_pred_4_nonrs[k2][1])), (255, 0, 255), 2)
        #img_orig_show = cv2.putText(img_orig_show, 'two', (int(pts_pred_4_nonrs[0][0]), int(pts_pred_4_nonrs[0][1])),
        #        cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

        errs_nonrs = np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_4_nonrs), -1))
        err_dict['error_stage_4_nonrs'] = errs_nonrs.tolist()

        M_4 = cv2.getPerspectiveTransform(np.float32(pts_pred_4_nonrs), np.float32(pts_in_warp))
        M_4_inv = cv2.getPerspectiveTransform(np.float32(pts_in_warp), np.float32(pts_pred_4_nonrs))
        img_patch_4_a = cv2.warpPerspective(img_orig, M_4, target_size, flags=cv2.INTER_CUBIC)
        img_pair_4 = np.stack((img_patch_4_a, img_pair[:, :, 1]), -1)

        pts_gt_4 = cv2.perspectiveTransform(np.float32(pts_in_orig_nonrs).reshape(-1,1,2), M_4)
        pts_gt_4 = pts_gt_4.reshape(-1, 2) - pts_in_warp
        pts_gt_4_norm = pts_gt_4 / np.array(target_size).reshape(-1, 2)

        # for generate next stage dataset
        img_pair_new = img_pair_4
        pts_new_norm = pts_gt_4_norm

        if np.max(np.abs(pts_new_norm[:, 0])) >= 0.5 or np.max(np.abs(pts_new_norm[:, 1])) >= 0.5:
            print('skip')
            flag = False

    return (img_pair_new, pts_new_norm, img_orig_show, err_dict, flag)


def main():
    model_dict = {}
    assert args.model_name in MODEL_MAP
    input_shape = (target_size[1], target_size[0], 2)
    model_1 = MODEL_MAP[args.model_name](weights_file=args.model_stage_1, input_shape=input_shape, 
            pooling=args.pooling, alpha=args.alpha)
    model_1.summary()
    model_dict['model_1'] = model_1
    if args.num_models >= 2:
        model_2 = MODEL_MAP[args.model_name](weights_file=args.model_stage_2, input_shape=input_shape, 
                pooling=args.pooling, alpha=args.alpha)
        model_2.summary()
        model_dict['model_2'] = model_2
    if args.num_models >=3:
        model_3 = MODEL_MAP[args.model_name](weights_file=args.model_stage_3, input_shape=input_shape, 
                pooling=args.pooling, alpha=args.alpha)
        model_3.summary()
        model_dict['model_3'] = model_3
    if args.num_models >=4:
        model_4 = MODEL_MAP[args.model_name](weights_file=args.model_stage_4, input_shape=input_shape, 
                pooling=args.pooling, alpha=args.alpha)
        model_4.summary()
        model_dict['model_4'] = model_4

    data_loader = loader(args.images_dir, args.num_samples_per_image)

    image_pair_new_list = []
    pts_new_list = []

    error_stage_1 = []
    # pts error in original image (before resize)
    error_stage_1_nonrs = []
    error_stage_2_nonrs = []
    error_stage_3_nonrs = []
    error_stage_4_nonrs = []

    count = 0
    if args.show == True:
        cv2.namedWindow('show', 0)
        cv2.namedWindow('img_pair', 0)

    for img_pair, pts_in_orig, pts_in_warp, img_orig, pts_in_orig_nonrs in data_loader:
        """
        img_pair: original image resized to (128, 128) + original warped image (128, 128)
        pts_in_orig: groundtruth points in resized original image (128, 128)
        pts_in_orig_nonrs: groundtruth points in non-resized image
        """
        count += 1
        print('{}'.format(count))

        img_pair_new, pts_new_norm, img_orig_show, err_dict, flag = process_image(img_pair, pts_in_orig, pts_in_warp, img_orig, pts_in_orig_nonrs, model_dict)
        error_stage_1 += err_dict['error_stage_1']
        error_stage_1_nonrs += err_dict['error_stage_1_nonrs']
        if 'error_stage_2_nonrs' in err_dict:
            error_stage_2_nonrs += err_dict['error_stage_2_nonrs']
        if 'error_stage_3_nonrs' in err_dict:
            error_stage_3_nonrs += err_dict['error_stage_3_nonrs']
        if 'error_stage_4_nonrs' in err_dict:
            error_stage_4_nonrs += err_dict['error_stage_4_nonrs']
        
        if count % 100 == 0:
            print('stage_1 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_1_nonrs)))
            if args.num_models >= 2:
                print('stage_2 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_2_nonrs)))
            if args.num_models >= 3:
                print('stage_3 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_3_nonrs)))
            if args.num_models >= 4:
                print('stage_4 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_4_nonrs)))


        if args.show == True:
            cv2.imshow('show', img_orig_show)
            cv2.imshow('img_pair', np.concatenate((img_pair_new[:, :, 0], img_pair_new[:, :, 1]), 1))
            print(pts_new_norm)
            key = cv2.waitKey(0)
            if key in [ord('q'), ord('Q')]:
                break

        if flag == True and args.gen_next_stage_data == True:
            image_pair_new_list.append(img_pair_new)
            pts_new_list.append(pts_new_norm.reshape(-1))

            if len(image_pair_new_list) >= 6400*2:
                pack(args.output_data_dir, image_pair_new_list, pts_new_list)
                image_pair_new_list = []
                pts_new_list = []

    if args.show == True:
        cv2.destroyAllWindows()

    if args.gen_next_stage_data == True:
        if len(image_pair_new_list) > 0:
            pack(args.output_data_dir, image_pair_new_list, pts_new_list)
            image_pair_new_list = []
            pts_new_list = []
    
    print('')
    print('*'*80)
    print('stage_1 | mean pts error in 128x128: {:.2f} [pixel]'.format(np.mean(error_stage_1)))
    print('stage_1 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_1_nonrs)))
    if args.num_models >= 2:
        print('stage_2 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_2_nonrs)))
    if args.num_models >= 3:
        print('stage_3 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_3_nonrs)))
    if args.num_models >= 4:
        print('stage_4 | mean pts error in origin image: {:.2f} [pixel]'.format(np.mean(error_stage_4_nonrs)))


if __name__ == '__main__':
    if args.debug == True:
        pdb.set_trace()
    
    if args.gen_next_stage_data == True:
        assert args.output_data_dir is not None
        assert not os.path.isdir(args.output_data_dir)
        os.makedirs(args.output_data_dir)
    
    main()