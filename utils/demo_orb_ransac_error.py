#coding=utf-8
import os, argparse
import cv2
import glob
import numpy as np
import pdb
import random

target_size = (128, 128)

def generate_points(image_size, img_orig=None):
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

    if img_orig is not None:
        img_patch_a = img_orig[y1:y2+1, x1:x2+1]
    else:
        img_patch_a = None

    pts = []
    # in order of [pt_tl, pt_tr, pt_bl, pt_br]
    pts_tmp = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for x, y in pts_tmp:
        perturb_x = random.randint(-bolder_w, bolder_w)
        perturb_y = random.randint(-bolder_h, bolder_h)
        pts.append((x+perturb_x, y+perturb_y))

    return np.array(pts), img_patch_a, pts_tmp

def warp(image, pts_in_orig, pts_in_warp, target_size):
    M = cv2.getPerspectiveTransform(np.float32(pts_in_orig), np.float32(pts_in_warp))
    return cv2.warpPerspective(image, M, target_size, flags=cv2.INTER_CUBIC)


def orb_sift_ransac(image_orig, image_warp, kp='sift'):
    """compute transformation matrix H from img_warp to img_orig
    """
    if kp == 'orb':
        print('ORB')
        detector = cv2.ORB_create() #cv2.xfeatures2d.ORB_create()
    elif kp == 'sift':
        print('SIFT')
        detector = cv2.xfeatures2d.SIFT_create()

    kp_warp = detector.detect(image_warp ,None)
    kp_warp, des_warp = detector.compute(image_warp, kp_warp)

    kp_orig = detector.detect(image_orig ,None)
    kp_orig, des_orig = detector.compute(image_orig, kp_orig)

    MIN_MATCH_COUNT = 10
    if des_warp is None or des_orig is None:
        return None, None  #np.identity(3)
    if len(des_warp) < MIN_MATCH_COUNT or len(des_orig) < MIN_MATCH_COUNT:
        return None, None

    M, matchesMask, img_match = None, None, None

    if 1:
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        if kp == 'sift':
            matches = bf.match(des_warp.astype(np.uint8), des_orig.astype(np.uint8))
        else:
            matches = bf.match(des_warp, des_orig)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        good = matches[:25] # we take the first 25 matches
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp_warp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([kp_orig[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(np.float32(des_warp), np.float32(des_orig), k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
            #good.append(m)
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp_warp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([kp_orig[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

    ###### tmp
    if matchesMask:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    #matchesMask = matchesMask, # draw only inliers
                    flags = 2)
        img_match = cv2.drawMatches(image_warp, kp_warp, image_orig, kp_orig, good, None,**draw_params)

    return M, img_match


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir_anns', type=str)
    parser.add_argument('--image_warp_gt_file', type=str)

    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--num_samples_per_img', type=int, default=1)

    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()

    if args.images_dir_anns is not None:
        # for crbot wall evaluate
        ann_file = os.path.join(args.images_dir_anns, 'annotation.txt')
        assert os.path.isfile(ann_file)
        ann_list = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                if line.strip() == '' or line.strip()[0] == '#':
                    continue
                anns = line.strip().split(' ')
                ann_list.append(anns)

        count = 0
        pts_err_list = []

        # 所有anns图片共享一个 image_warp
        assert os.path.isfile(args.image_warp_gt_file)
        image_warp = cv2.imread(args.image_warp_gt_file)
        h, w = image_warp.shape[:2]
        pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)

        for anns in ann_list:
            count += 1
            print(count)

            image_name = os.path.join(args.images_dir_anns, anns[0])
            image_orig = cv2.imread(image_name, 0)
            # pts order: [p_tl, p_tr, p_bl, p_br]
            pts_in_orig = np.array([int(t) for t in anns[1:]]).reshape(-1, 2)

            """
            w = np.max(pts_in_orig[:, 0]) - np.min(pts_in_orig[:, 0]) + 1
            h = np.max(pts_in_orig[:, 1]) - np.min(pts_in_orig[:, 1]) + 1
            pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)
            image_warp = warp(image_orig, pts_in_orig, pts_in_warp, (w, h))
            """

            M_warp2orig, img_match_warp2orig = orb_sift_ransac(image_orig, image_warp)
            if M_warp2orig is not None:
                pts_pred_nonrs = cv2.perspectiveTransform(np.float32(pts_in_warp).reshape(-1,1,2), M_warp2orig)
                pts_pred_nonrs = pts_pred_nonrs.reshape(-1, 2)
                # use image_orig to clip
                pts_pred_nonrs[:, 0] = np.clip(pts_pred_nonrs[:, 0], 0, image_orig.shape[1])
                pts_pred_nonrs[:, 1] = np.clip(pts_pred_nonrs[:, 1], 0, image_orig.shape[0])
            
                err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig - pts_pred_nonrs), -1)))
                print(err)
                pts_err_list.append(err)

                image_orig_show = cv2.cvtColor(image_orig, cv2.COLOR_GRAY2BGR)
                for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                    image_orig_show = cv2.line(image_orig_show, (pts_in_orig[k1][0], pts_in_orig[k1][1]),
                            (pts_in_orig[k2][0], pts_in_orig[k2][1]), (0, 255, 0), 2)
                
                img_warp2orig_res = np.copy(image_orig_show)
                for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                    img_warp2orig_res = cv2.line(img_warp2orig_res, (pts_pred_nonrs[k1][0], pts_pred_nonrs[k1][1]),
                            (pts_pred_nonrs[k2][0], pts_pred_nonrs[k2][1]), (0, 0, 255), 2)

                
                assert os.path.isdir('/Users/jianchong/output/tmp')
                #cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_img_warp.png'.format(count), image_warp)
                #cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_img_gt.png'.format(count), image_orig_show)
                cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_match_warp2orig.png'.format(count), img_match_warp2orig)
                cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_res_warp2orig.png'.format(count), img_warp2orig_res)
                

                ##### vis #####
                if args.show == True:
                    cv2.imshow('show', img_warp2orig_res)
                    key = cv2.waitKey(0)
                    if key in [ord('q'), ord('Q')]:
                        break

        print('')
        print('total sample: {}'.format(count))
        print('mean pts error between image_orig and image_warp: {:.2f} [pixel]'.format(np.mean(pts_err_list)))
        print('num match between image_orig and image_warp {}'.format(len(pts_err_list)))

    elif args.images_dir is not None:
        pts_err_image_a = []
        pts_err_image_patch_a = []
        count = 0
        flag_end = False
        w, h = target_size
        for img_file in glob.glob(os.path.join(args.images_dir, '*.jpg')):
            if flag_end == True:
                break
            for _ in range(args.num_samples_per_img):
                img_orig = cv2.imread(img_file, 0)
                if img_orig.shape[0] < target_size[1]*3//2 or img_orig.shape[1] < target_size[0]*3//2:
                    continue
                count += 1

                pts_in_orig_nonrs, img_patch_a, pts_tmp = generate_points(img_orig.shape[:2][::-1], img_orig)
                pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)
                img_warp = warp(img_orig, pts_in_orig_nonrs, pts_in_warp, target_size)

                image_orig_show = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
                for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                    image_orig_show = cv2.line(image_orig_show, (pts_in_orig_nonrs[k1][0], pts_in_orig_nonrs[k1][1]),
                            (pts_in_orig_nonrs[k2][0], pts_in_orig_nonrs[k2][1]), (0, 255, 0), 2)
                
                M_warp2orig, img_match_warp2orig = orb_sift_ransac(img_orig, img_warp)
                if M_warp2orig is not None:
                    pts_pred_nonrs = cv2.perspectiveTransform(np.float32(pts_in_warp).reshape(-1,1,2), M_warp2orig)
                    pts_pred_nonrs = pts_pred_nonrs.reshape(-1, 2)
                    # use image_orig to clip
                    pts_pred_nonrs[:, 0] = np.clip(pts_pred_nonrs[:, 0], 0, img_orig.shape[1])
                    pts_pred_nonrs[:, 1] = np.clip(pts_pred_nonrs[:, 1], 0, img_orig.shape[0])

                    err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_nonrs), -1)))
                    print(err)
                    pts_err_image_a.append(err)

                    pts_pred_nonrs = pts_pred_nonrs.astype(np.int32)
                    img_warp2orig_res = np.copy(image_orig_show)
                    for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_warp2orig_res = cv2.line(img_warp2orig_res, (pts_pred_nonrs[k1][0], pts_pred_nonrs[k1][1]),
                                (pts_pred_nonrs[k2][0], pts_pred_nonrs[k2][1]), (0, 0, 255), 2)
                    
                    assert os.path.isdir('/Users/jianchong/output/tmp')
                    cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_img_warp.png'.format(count), img_warp)
                    cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_img_gt.png'.format(count), image_orig_show)

                    cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_match_warp2orig.png'.format(count), img_match_warp2orig)
                    cv2.imwrite('/Users/jianchong/output/tmp/'+'img_{}_res_warp2orig.png'.format(count), img_warp2orig_res)
                    
                
                """
                M_warp2patch_a, img_match_warp2patch_a = orb_sift_ransac(img_patch_a, img_warp)
                if M_warp2patch_a is not None:
                    pts_pred_patch_a = cv2.perspectiveTransform(np.float32(pts_in_warp).reshape(-1,1,2), M_warp2patch_a)
                    pts_pred_patch_a_offset = pts_pred_patch_a.reshape(-1, 2) - np.array(pts_in_warp).reshape(-1, 2) 
                    pts_pred_patch_a_offset = np.clip(pts_pred_patch_a_offset, -32, 32)
                    
                    pts_pred_patch_a_nonrs = pts_pred_patch_a_offset + np.array(pts_tmp).reshape(-1, 2)

                    err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_patch_a_nonrs), -1)))
                    print(err)
                    pts_err_image_patch_a.append(err)

                    pts_pred_patch_a_nonrs = pts_pred_patch_a_nonrs.astype(np.int32)
                    img_warp2patch_a_res = np.copy(image_orig_show)
                    for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                        img_warp2patch_a_res = cv2.line(img_warp2patch_a_res, (pts_pred_patch_a_nonrs[k1][0], pts_pred_patch_a_nonrs[k1][1]),
                                (pts_pred_patch_a_nonrs[k2][0], pts_pred_patch_a_nonrs[k2][1]), (200, 0, 255), 2)
                """
 
                ##### vis #####
                if args.show == True:
                    cv2.imshow('img_warp2orig_res', img_warp2orig_res)
                    if img_match_warp2orig is not None:
                        cv2.imshow('img_match_warp2orig', img_match_warp2orig)
                    
                    #cv2.imshow('img_warp2patch_a_res', img_warp2patch_a_res)
                    #if img_match_warp2patch_a is not None:
                    #    cv2.imshow('img_match_warp2patch_a', img_match_warp2patch_a)

                    key = cv2.waitKey(0)
                    if key in [ord('q'), ord('Q')]:
                        flag_end = True
                        break
                
        if args.show == True:
            cv2.destroyAllWindows()

        print('')
        assert os.path.isdir('/Users/jianchong/output/tmp')
        with open('/Users/jianchong/output/tmp/' + 'pts_err_warp2orig.txt', 'w') as f:
            for item in pts_err_image_a:
                f.write("%s\n" % item)

        print('total sample: {}'.format(count))
        print('mean pts error between image_orig and image_warp: {:.2f} [pixel]'.format(np.mean(pts_err_image_a)))
        print('num match between image_orig and image_warp {}'.format(len(pts_err_image_a)))
        #print('')
        #print('mean pts error between image_patch_a and image_warp: {:.2f} [pixel]'.format(np.mean(pts_err_image_patch_a)))
        #print('num match between image_patch_a and image_warp {}'.format(len(pts_err_image_patch_a)))