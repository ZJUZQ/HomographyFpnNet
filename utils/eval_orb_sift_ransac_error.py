#coding=utf-8
import os, argparse
import cv2
import glob
import numpy as np
import pdb
import random

target_size = (128, 128)

def generate_points(image_size):
    """
    image_size: (w, h)
    """
    w, h = image_size
    # step 1: 确定cropped rectangle size, 预留 1/4 扰动
    h_ = random.randint(128, h * 2 // 3)
    w_ = random.randint(128, w * 2 // 3)

    # step 2: select random rectangle
    x1 = random.randint(w_//4, w - w_ - w_//4)
    y1 = random.randint(h_//4, h - h_ - h_//4)
    x2 = x1 + w_ - 1
    y2 = y1 + h_ - 1

    pts = []
    # in order of [pt_tl, pt_tr, pt_bl, pt_br]
    pts_tmp = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for x, y in pts_tmp:
        x += random.randint(-w_//4+1, w_//4-1)
        y += random.randint(-h_//4+1, h_//4-1)
        assert x >= 0 and x < w
        assert y >= 0 and y < h
        pts.append((x, y))
    return np.array(pts)

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
        draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                    singlePointColor = None,
                    #matchesMask = matchesMask, # draw only inliers
                    flags = 2)
        img_match = cv2.drawMatches(image_warp, kp_warp, image_orig, kp_orig, good, None,**draw_params)

    return M, img_match


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('num_samples_per_img', type=int)
    parser.add_argument('num_images', type=int)

    parser.add_argument('--detector', type=str, default='orb')

    parser.add_argument('--save_res', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()
    
    assert args.detector in ['orb', 'sift']

    pts_err_list = []

    num_image_count = 0
    count = 0
    w, h = target_size
    for img_file in glob.glob(os.path.join(args.images_dir, '*.jpg')):
        if num_image_count >= args.num_images:
            break
        img_orig = cv2.imread(img_file, 0)
        if img_orig.shape[0] < target_size[1]*3//2 or img_orig.shape[1] < target_size[0]*3//2:
            continue
        num_image_count += 1

        for _ in range(args.num_samples_per_img):
            count += 1
            #pts_in_orig_nonrs = np.array([[60, 70], [234, 70], [70, 300], [221, 310]])
            #print('use fix points for tmp')
            pts_in_orig_nonrs = generate_points(img_orig.shape[:2][::-1])
            pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)
            img_warp = warp(img_orig, pts_in_orig_nonrs, pts_in_warp, target_size)

            image_orig_show = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
            for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                image_orig_show = cv2.line(image_orig_show, (pts_in_orig_nonrs[k1][0], pts_in_orig_nonrs[k1][1]),
                        (pts_in_orig_nonrs[k2][0], pts_in_orig_nonrs[k2][1]), (0, 255, 0), 2)
            
            M_warp2orig, img_match_warp2orig = orb_sift_ransac(img_orig, img_warp, kp=args.detector)
            if M_warp2orig is not None:
                pts_pred_nonrs = cv2.perspectiveTransform(np.float32(pts_in_warp).reshape(-1,1,2), M_warp2orig)
                pts_pred_nonrs = pts_pred_nonrs.reshape(-1, 2)
                # use image_orig to clip
                pts_pred_nonrs[:, 0] = np.clip(pts_pred_nonrs[:, 0], 0, img_orig.shape[1])
                pts_pred_nonrs[:, 1] = np.clip(pts_pred_nonrs[:, 1], 0, img_orig.shape[0])

                err = np.mean(np.sqrt(np.sum(np.square(pts_in_orig_nonrs - pts_pred_nonrs), -1)))
                print(err)
                pts_err_list.append(err)

                pts_pred_nonrs = pts_pred_nonrs.astype(np.int32)
                img_warp2orig_res = np.copy(image_orig_show)
                for k1, k2 in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                    img_warp2orig_res = cv2.line(img_warp2orig_res, (pts_pred_nonrs[k1][0], pts_pred_nonrs[k1][1]),
                            (pts_pred_nonrs[k2][0], pts_pred_nonrs[k2][1]), (0, 0, 255), 2)
                    
                    img_match_warp2orig = cv2.line(img_match_warp2orig, (pts_in_orig_nonrs[k1][0]+128, pts_in_orig_nonrs[k1][1]),
                            (pts_in_orig_nonrs[k2][0]+128, pts_in_orig_nonrs[k2][1]), (0, 255, 0), 2)
                    img_match_warp2orig = cv2.line(img_match_warp2orig, (pts_pred_nonrs[k1][0]+128, pts_pred_nonrs[k1][1]),
                            (pts_pred_nonrs[k2][0]+128, pts_pred_nonrs[k2][1]), (0, 0, 255), 2)
                cv2.putText(img_match_warp2orig, 'mean corner error: {:.1f}'.format(err), (140, 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,0,255), 1)
                    
                if args.save_res == True:
                    #cv2.imwrite(os.path.join(args.output_dir + 'img_{}_img_warp.png'.format(num_image_count)), img_warp )
                    #cv2.imwrite(os.path.join(args.output_dir + 'img_{}_img_gt.png'.format(num_image_count)), image_orig_show)
                    cv2.imwrite(os.path.join(args.output_dir + 'img_{}_match_warp2orig_{}.png'.format(num_image_count, args.detector)), img_match_warp2orig)
                    #cv2.imwrite(os.path.join(args.output_dir + 'img_{}_res_warp2orig.png'.format(num_image_count)), img_warp2orig_res)
                    
    print('')
    print('*'*80)
    outFile = os.path.join(args.images_dir, '..', 'pts_err_{}.txt'.format(args.detector))
    outFile = os.path.abspath(outFile)
    print('save errors to file: {}'.format(outFile))
    with open(outFile, 'w') as f:
        for item in pts_err_list:
            f.write("%s\n" % item)
    
    print('total sample: {}'.format(count))
    print('mean pts error between image_orig and image_warp: {:.2f} [pixel]'.format(np.mean(pts_err_list)))
    print('num match between image_orig and image_warp {}'.format(len(pts_err_list)))