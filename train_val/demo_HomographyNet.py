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
from dataset.generate_dataset_homographynet import warp, scale_down, center_crop, crop

MODEL_MAP = {'vgg': create_homographynet_vgg,
            'vgg_fpn': create_homographynet_vgg_fpn,
            'mobilenet_v2': create_homographynet_mobilenet_v2,
            }

image_size = (320, 240) # (w, h)
target_size = (128, 128)
model_input_shape = (128, 128)
pts_norm_value = (32, 32)

def generate_points_test():
    """
    Choose top-left corner of patch (assume 0,0 is top-left of image)
    Restrict points to within 12-px from the border, 并且考虑到扰动范围[-p, p]
    for x of top-left:
        x = [12 + p, 320 - 12 - p - 128]
        y = [12 + p, 240 - 12 - p - 128]
    """
    p = target_size[0] // 4
    x = random.randint(12 + p, image_size[0] - 12 - p - target_size[0]) # [12 + p, 320 - 12 - p - 128]
    y = random.randint(12 + p, image_size[1] - 12 - p - target_size[1])  # 12 + p, 240 - 12 - p - 128]

    # pts in ordere [pt_tl, pt_tr, pt_bl, pt_br]
    patch = [
        (x, y),
        (x + target_size[0], y),
        (x, y + target_size[1]),
        (x + target_size[0], y + target_size[1])
    ]

    # Perturb
    perturbed_patch = [(x + random.randint(-p, p), y + random.randint(-p, p)) for x, y in patch]
    return np.array(patch), np.array(perturbed_patch)


def loader(data_dir, num_samples_per_image, num_img=5000):
    
    count = 0
    for image_path in glob.iglob(os.path.join(data_dir, '*.jpg')):
        if count > num_img:
            return 

        img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_orig.shape[:2][::-1] < image_size:
            continue
        count += 1
        img_a = scale_down(img_orig, image_size)
        img_a = center_crop(img_a, image_size)
        assert img_a.shape[::-1] == image_size

        for _ in range(num_samples_per_image):

            pts_in_warp, pts_in_orig = generate_points_test()
            img_b = warp(img_a, pts_in_orig, pts_in_warp, image_size)

            patch_a = crop(img_a, pts_in_warp[0], target_size)
            patch_b = crop(img_b, pts_in_warp[0], target_size)
            yield img_a, patch_a, pts_in_orig, img_b, patch_b, pts_in_warp


def predict_pts_in_orig(img_a, img_patch_a, img_b, img_patch_b, pts_in_warp, model_dict):

    pts_pred_nonrs_list = []

    img_pair = np.stack((img_patch_a, img_patch_b), -1)

    img_pair_norm = (img_pair - 127.5) / 127.5
    pts_pred_1 = model_dict['model_1'].predict(img_pair_norm[np.newaxis])
    pts_pred_1_nonrs = pts_pred_1.reshape(-1, 2) * np.array(pts_norm_value).reshape(-1, 2) + pts_in_warp.reshape(-1, 2)
    
    pts_pred_nonrs_list.append(pts_pred_1_nonrs)

    return pts_pred_nonrs_list


def main(args):
    input_shape = (model_input_shape[1], model_input_shape[0], 2) # e.g., (128, 128, 2)

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

    pts_err_list = []
    count = 0

    if args.show == True:
        cv2.namedWindow('show', 0)
    
    if args.img_dir is not None:
        data_loader = loader(args.img_dir, args.num_samples_per_image)
        for img_a, img_patch_a, pts_in_orig, img_b, img_patch_b, pts_in_warp in data_loader:
            count += 1
            print(count)
            assert img_patch_a.shape[::-1] == target_size # (256, 256)
            assert img_patch_b.shape[::-1] == target_size

            pts_pred_nonrs_list = predict_pts_in_orig(img_a, img_patch_a, img_b, img_patch_b, pts_in_warp, model_dict)
            pts_err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig - pts_pred_nonrs_list[-1]), -1)))
            pts_err_list.append(pts_err)
            print(pts_in_orig)
            print(pts_pred_nonrs_list[-1])
            print(pts_err)

            if args.show == True:
                img_show = np.ones((img_a.shape[0]*len(pts_pred_nonrs_list), img_a.shape[1]*2, 3), np.uint8)*255
                for k in range(len(pts_pred_nonrs_list)):
                    img_a_show = np.copy(img_a)
                    img_a_show = cv2.cvtColor(img_a_show, cv2.COLOR_GRAY2BGR)
                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_a_show = cv2.line(img_a_show, (int(pts_in_orig[k1][0]), int(pts_in_orig[k1][1])),
                                    (int(pts_in_orig[k2][0]), int(pts_in_orig[k2][1])), (0, 255, 0), 2)

                    img_b_show = np.copy(img_b)
                    img_b_show = cv2.cvtColor(img_b_show, cv2.COLOR_GRAY2BGR)
                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_b_show = cv2.line(img_b_show, (int(pts_in_warp[k1][0]), int(pts_in_warp[k1][1])),
                                    (int(pts_in_warp[k2][0]), int(pts_in_warp[k2][1])), (0, 255, 0), 2)
                    
                    cv2.putText(img_a_show, 'Mean corner error: {:.1f}'.format(pts_err), (10, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)
                    
                    #assert os.path.isdir('/Users/jianchong/output/tmp/')             
                    #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_image_gt.png'.format(count)), img_orig_show)
                    #cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_image_warp.png'.format(count)), img_warp)

                    for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_a_show = cv2.line(img_a_show, (int(pts_pred_nonrs_list[k][k1][0]), int(pts_pred_nonrs_list[k][k1][1])),
                                    (int(pts_pred_nonrs_list[k][k2][0]), int(pts_pred_nonrs_list[k][k2][1])), (0, 0, 255), 2)
                    
                    img_show[k*img_a_show.shape[0]:(k+1)*img_a_show.shape[0], :img_a_show.shape[1], :] = img_a_show
                    img_show[k*img_b_show.shape[0]:(k+1)*img_b_show.shape[0], -img_b_show.shape[1]: :] = img_b_show
                
                cv2.imshow('show', img_show)
                key = cv2.waitKey(0)
                if key in [ord('q'), ord('Q')]:
                    return
                if key in [ord('s')]:
                    cv2.imwrite(os.path.join('/Users/jianchong/output/tmp/', '{}_{}_res.png'.format(args.model_name, count)), img_show)

            if count % 50 == 0:
                print('mean pts error of {} images: {:.2f} [pixel]'.format(count, np.mean(pts_err_list)))

    print('')
    print('*'*80)
    print('mean pts error: {:.2f} [pixel]'.format(np.mean(pts_err_list)))
    print('num samples: {}'.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_models', type=int, help='number of modules')
    parser.add_argument('--model_stage_1', type=str)

    parser.add_argument('--model_stage_2', type=str)
    parser.add_argument('--model_stage_3', type=str)
    parser.add_argument('--model_stage_4', type=str)
    parser.add_argument('--model_name', type=str, default='vgg')

    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--num_samples_per_image', type=int, default=2)

    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()
    main(args)
    if args.show == True:
        cv2.destroyAllWindows()