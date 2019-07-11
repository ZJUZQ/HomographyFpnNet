#coding=utf-8
"""
作用：
"""
import os
import copy
import pdb
import glob 
import argparse
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.linear_assignment_ import linear_assignment

def warp(img, pts_in_orig, pts_in_warp, target_size):
    """
    """
    M = cv2.getPerspectiveTransform(np.float32(pts_in_orig), np.float32(pts_in_warp))
    return cv2.warpPerspective(img, M, target_size, flags=cv2.INTER_CUBIC)


class CoordinateStore(object):
    def __init__(self):
        self.points = []
        self.drawing = False
    
    def select_point(self, event, x, y, flags, param):
        # mouse callback function
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                #cv2.circle(img, (x, y), 2, (0, 0, 255), 5)
                self.points.append([x, y])


def get_geometric_transform_pts(image):
    coordinate_storer = CoordinateStore()

    cv2.namedWindow('GetPoints', 0)
    cv2.setMouseCallback('GetPoints', coordinate_storer.select_point)

    while(1):
        img = copy.copy(image)
        for p in coordinate_storer.points:
            cv2.circle(img, tuple(p), radius=10, color=(0, 0, 255), thickness=2)
        cv2.putText(img, text='{} points selected'.format(len(coordinate_storer.points)), 
                    org=(20, 40), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow('GetPoints', img)
        k = cv2.waitKey(1) & 0xFF
        if k in [27, ord('q'), ord('Q')]:
            break
        if k in [ord('d'), ord('D')]:
            coordinate_storer.points = coordinate_storer.points[:-1]

    cv2.destroyWindow('GetPoints')
    pts = coordinate_storer.points
    assert len(pts) == 4
    x_min, x_max, y_min, y_max = (np.min(np.array(pts)[:, 0]), np.max(np.array(pts)[:, 0]),
            np.min(np.array(pts)[:, 1]), np.max(np.array(pts)[:, 1]))
    
    # [p_tl, p_tr, p_bl, p_br]
    pts_bbox = [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
    educ_cost = euclidean_distances(np.array(pts_bbox), np.array(pts))
    matched_indices = linear_assignment(educ_cost)

    pts_order = [] # [p_tl, p_tr, p_bl, p_br]
    for i in range(4):
        p_id = matched_indices[np.where(matched_indices[:, 0] == i)[0], 1]
        pts_order.append(pts[int(p_id)])
    return pts_order


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--check', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()

    image_files = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    image_dir = args.image_dir

    ann_list = []
    if args.check == True:
        ann_file = os.path.join(args.image_dir, 'annotation.txt')
        with open(ann_file, 'r') as f:
            for l in f.readlines():
                if l.strip() == '' or l[0] == '#':
                    continue
                ann_list.append(l.strip().split(' '))
    else:
        for img_file in image_files:
            import re
            num = int(re.search('img_([0-9]+).jpg', os.path.basename(img_file)).groups()[0])
            if num <= 50:
                continue
                
            pts_in_orig = get_geometric_transform_pts(cv2.imread(img_file))

            pts_in_orig = np.array(pts_in_orig, dtype=np.int32).reshape(-1).tolist()
            ann_list.append([os.path.basename(img_file)] + pts_in_orig)
            print(' '.join([str(t) for t in ann_list[-1]]))
    
    for ann in ann_list:
        file_name = ann[0]
        image_orig = cv2.imread(os.path.join(image_dir, file_name), 0)
        pts_in_orig = np.array([float(t) for t in ann[1:]]).reshape(-1, 2)
        w = int(np.max(pts_in_orig[:, 0]) - np.min(pts_in_orig[:, 0]) + 1)
        h = int(np.max(pts_in_orig[:, 1]) - np.min(pts_in_orig[:, 1]) + 1)

        pts_in_warp = np.array([0, 0, w-1, 0, 0, h-1, w-1, h-1]).reshape(-1, 2)
        image_warp = warp(image_orig, pts_in_orig, pts_in_warp, (w, h))
        cv2.imwrite(os.path.join(image_dir, os.path.splitext(file_name)[0]+'_warp.png'), image_warp)

    print('')
    print('*'*80)
    for info in ann_list:
        print(' '.join(str(item) for item in info))

    for info in ann_list:
        for (i, t) in enumerate(info):
            if i > 0:
                info[i] =  int(t)
        image = cv2.imread(os.path.join(image_dir, info[0]))
        cv2.circle(image, (info[1], info[2]), 5, (0, 0, 255), 3) # p_tl
        cv2.circle(image, (info[3], info[4]), 5, (0, 0, 255), 3) # p_tr
        cv2.circle(image, (info[5], info[6]), 5, (255, 0, 0), 3) # p_bl
        cv2.circle(image, (info[7], info[8]), 5, (255, 0, 0), 3) # p_br
        cv2.line(image, (info[1], info[2]), (info[3], info[4]), (0, 255, 0), 2)  
        cv2.line(image, (info[3], info[4]), (info[7], info[8]), (0, 255, 0), 2)  
        cv2.line(image, (info[7], info[8]), (info[5], info[6]), (0, 255, 0), 2)  
        cv2.line(image, (info[5], info[6]), (info[1], info[2]), (0, 255, 0), 2) 
        cv2.imshow('show', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows() 