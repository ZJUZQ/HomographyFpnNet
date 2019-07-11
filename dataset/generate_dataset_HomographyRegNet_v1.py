#coding=utf-8
import random
import glob
import os
import uuid

from queue import Queue, Empty
from threading import Thread

import numpy as np
import cv2
import pdb
import math
import argparse
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.linear_assignment_ import linear_assignment

"""
Example usage:
    python dataset/generate_dataset_v2.py ~/dataset/pascal_voc/VOCdevkit/VOC2012/JPEGImages/ ~/dataset/homography_voc_dataset/train_240x320_voc12 5
"""
TARGET_SIZE = (128, 128) # (w, h)


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
    for x, y in [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]:
        x += random.randint(-w_//4+1, w_//4-1)
        y += random.randint(-h_//4+1, h_//4-1)
        assert x >= 0 and x < w
        assert y >= 0 and y < h
        pts.append((x, y))
    return np.array(pts)

def warp(img, pts_in_orig, pts_in_warp, target_size):
    """
    """
    M = cv2.getPerspectiveTransform(np.float32(pts_in_orig), np.float32(pts_in_warp))
    return cv2.warpPerspective(img, M, target_size, flags=cv2.INTER_CUBIC)


def process_image(image_path, num_output=1):
    """
    为了保证分辨率，我们首先对原图进行pts 选择和warp，之后resize原图
    """

    # Read as grayscale
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    target_size = TARGET_SIZE
    # 允许 1/4 扰动
    if img_orig.shape[0] < target_size[1] * 3 // 2 or img_orig.shape[1] < target_size[0] * 3 // 2:
        return

    image_pairs = []
    pts_norm_gts = []

    # in order of [pt_tl, pt_tr, pt_bl, pt_br]
    pts_in_warp = np.array([(0, 0), (target_size[0]-1, 0), (0, target_size[1]-1), (target_size[0]-1, target_size[1]-1)])

    while len(pts_norm_gts) < num_output:
        pts_in_orig = generate_points(img_orig.shape[:2][::-1])
        img_warp = warp(img_orig, pts_in_orig, pts_in_warp, target_size)
        img = cv2.resize(img_orig, target_size, interpolation=cv2.INTER_CUBIC)
        pts_in_orig_norm = pts_in_orig.reshape(-1, 2) / np.array(img_orig.shape[:2][::-1]).reshape(1, 2)

        try:
            d = np.stack((img, img_warp), axis=-1)
        except ValueError:
            continue
        image_pairs.append(d)
        pts_norm_gts.append(pts_in_orig_norm.reshape(-1))

    print('done:', image_path)
    return image_pairs, pts_norm_gts


class Worker(Thread):
    def __init__(self, input_queue, output_queue, num_samples_per_img):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_samples_per_img = num_samples_per_img # number samples to generate with one original image

    def run(self):
        while True:
            img_path = self.input_queue.get()
            if img_path is None:
                break
            output = process_image(img_path, self.num_samples_per_img)
            self.input_queue.task_done()
            if output is not None:
                self.output_queue.put(output)


def pack(outdir, image_pairs, pts_norm_gts):
    name = str(uuid.uuid4())
    pack = os.path.join(outdir, name + '.npz')
    with open(pack, 'wb') as f:
        np.savez(f, images=np.stack(image_pairs), pts=np.stack(pts_norm_gts))
    print('bundled:', name)


def bundle(queue, outdir):
    image_pairs = []
    pts_norm_gts = []
    #orig_points = []
    #perturbed_points = []
    while True:
        try:
            d, o = queue.get(timeout=60) # seconds
        except Empty:
            break
        image_pairs.extend(d)
        pts_norm_gts.extend(o)

        if len(image_pairs) >= 6400*2:
            pack(outdir, image_pairs, pts_norm_gts)
            image_pairs = []
            pts_norm_gts = []
        queue.task_done()

    if image_pairs:
        pack(outdir, image_pairs, pts_norm_gts)


def test():
    print('*'*80)
    print('*'*10 + '  TEST  ' + '*'*10)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_samples_per_img', type=int)

    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    num_samples_per_img = args.num_samples_per_img
    input_dir = args.input_dir

    for i in glob.iglob(os.path.join(input_dir, '*.jpg')):
        image_pairs, pts_norm_gts = process_image(i, num_samples_per_img)
        for k in range(len(image_pairs)):
            img_orig = image_pairs[k][:, :, 0]
            img_orig = np.ascontiguousarray(img_orig, dtype=np.uint8)
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
            img_warp = image_pairs[k][:, :, 1]
            img_warp = np.ascontiguousarray(img_warp, dtype=np.uint8)
            img_warp = cv2.cvtColor(img_warp, cv2.COLOR_GRAY2BGR)
            pts_in_orig = np.array(pts_norm_gts[k]).reshape(-1, 2) * np.array(img_orig.shape[:2][::-1]).reshape(1, -1)
            for p in pts_in_orig:
                img_orig = cv2.circle(img_orig, (int(p[0]), int(p[1])), 5, (0, 0, 255), 2)
            pts_in_orig = np.array(pts_in_orig, dtype=np.int32)
            for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                img_orig = cv2.line(img_orig, (pts_in_orig[k1][0], pts_in_orig[k1][1]), (pts_in_orig[k2][0], pts_in_orig[k2][1]), (0, 0, 255), 2)

            cv2.imshow('show', np.concatenate((img_orig, img_warp), 1))
            key = cv2.waitKey(0)
            if key in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_samples_per_img', type=int)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug == True:
        pdb.set_trace()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    num_samples_per_img = args.num_samples_per_img
    input_dir = args.input_dir

    # Create a queue to communicate with the worker threads
    input_queue = Queue()
    output_queue = Queue()

    num_workers = args.num_workers
    workers = []
    # Create worker threads
    for i in range(num_workers):
        worker = Worker(input_queue, output_queue, num_samples_per_img)
        worker.start()
        workers.append(worker)

    for i in glob.iglob(os.path.join(input_dir, '*.jpg')):
        input_queue.put(i)

    bundle(output_queue, output_dir)

    input_queue.join()
    for i in range(num_workers):
        input_queue.put(None)
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    main()
    #test()