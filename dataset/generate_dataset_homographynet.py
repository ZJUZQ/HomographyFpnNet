#coding=utf-8
"""
dataset generateion for "Homography Estimation from Image Pairs with Hierarchical Convolutional Networks"
"""
import sys
import random
import glob
import os.path
import uuid

from queue import Queue, Empty
from threading import Thread

import numpy as np
import cv2
import pdb
import argparse


def scale_down(img, target_size):
    # 保持长宽比的缩放， return_img.shape >= target_size
    src_height, src_width = img.shape
    src_ratio = src_height/src_width
    target_width, target_height = target_size
    if src_ratio < target_height/target_width:
        dst_size = (int(np.round(target_height/src_ratio)), target_height)
    else:
        dst_size = (target_width, int(np.round(target_width*src_ratio)))
    return cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)

def crop(img, origin, size):
    width, height = size
    x, y = origin
    return img[y:y + height, x:x + width]

def center_crop(img, target_size):
    target_width, target_height = target_size
    # Note the reverse order of width and height
    height, width = img.shape
    x = int(np.round((width - target_width)/2))
    y = int(np.round((height - target_height)/2))
    return crop(img, (x, y), target_size)

def generate_points():
    """
    Choose top-left corner of patch (assume 0,0 is top-left of image)
    Restrict points to within 12-px from the border, 并且考虑到扰动范围[-p, p]
    for x of top-left:
        x = [12 + p, 320 - 12 - p - 128]
        y = [12 + p, 240 - 12 - p - 128]
    """
    p = 32
    x = random.randint(44, 148) # [12 + p, 320 - 12 - p - 128]
    y = random.randint(44, 68)  # 12 + p, 240 - 12 - p - 128]
    # pts in ordere [pt_tl, pt_tr, pt_bl, pt_br]
    patch = [
        (x, y),
        (x + 128, y),
        (x, y + 128),
        (x + 128, y + 128)
    ]

    # Perturb
    perturbed_patch = [(x + random.randint(-p, p), y + random.randint(-p, p)) for x, y in patch]
    return np.array(patch), np.array(perturbed_patch)

def warp(img_src, pts_src, pts_dst, target_size):
    """
    """
    M = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
    img_dst = cv2.warpPerspective(img_src, M, target_size, flags=cv2.INTER_CUBIC)
    return img_dst


def process_image(image_path, num_output=1):
    """
    Train set:
        1.  All images are resized to 320x240 and converted to grayscale. We then 
            generate 500,000 pairs of image patches sized 128x128 related by a homography.
        2. We choose ρ = 32, which means that each corner of the 128x128 grayscale image can be
            perturbed by a maximum of one quarter of the total image edge size. 
    Test set:
        1. randomly chose 5000 images from the test set and resized each image to grayscale
            640x480, and generate a pairs of image patches sized 256x256x2 and corresponding 
            ground truth homography, with ρ = 64.
    """
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    target_size = (320, 240) # (w, h)
    patch_size = (128, 128)

    if img.shape[:2][::-1] < target_size:
        return

    img_a = scale_down(img, target_size)
    img_a = center_crop(img_a, target_size)
    assert img_a.shape[::-1] == target_size
    
    image_pairs = []
    offsets = []

    while len(offsets) < num_output:
        pts_orig, pts_perturbed = generate_points()
        patch_a = crop(img_a, pts_orig[0], patch_size)
        img_b = warp(img_a, pts_perturbed, pts_orig, target_size)
        patch_b = crop(img_b, pts_orig[0], patch_size)
        try:
            d = np.stack((patch_a, patch_b), axis=-1)
        except ValueError:
            continue
        image_pairs.append(d)
        offset = (pts_perturbed - pts_orig).reshape(-1)
        offsets.append(offset)

    print('done:', image_path)
    return image_pairs, offsets


class Worker(Thread):
    def __init__(self, input_queue, output_queue, num_samples):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_samples = num_samples # number samples to generate with one original image

    def run(self):
        while True:
            img_path = self.input_queue.get()
            if img_path is None:
                break
            output = process_image(img_path, self.num_samples)
            self.input_queue.task_done()
            if output is not None:
                self.output_queue.put(output)

def pack(outdir, image_pairs, offsets):
    name = str(uuid.uuid4())
    pack = os.path.join(outdir, name + '.npz')
    with open(pack, 'wb') as f:
        np.savez(f, images=np.stack(image_pairs), offsets=np.stack(offsets))
    print('bundled:', name)


def bundle(queue, outdir):
    image_pairs = []
    offsets = []

    while True:
        try:
            d, o = queue.get(timeout=60) # seconds
        except Empty:
            break
        image_pairs.extend(d)
        offsets.extend(o)
        #orig_points.extend(orig)
        #perturbed_points.extend(perturbed)

        if len(image_pairs) >= 6400*2:
            pack(outdir, image_pairs, offsets)
            image_pairs = []
            offsets = []
        queue.task_done()

    if image_pairs:
        pack(outdir, image_pairs, offsets)


def main(args):
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    num_samples_per_img = args.num_samples_per_img
    input_dir = args.input_dir

    # Create a queue to communicate with the worker threads
    input_queue = Queue()
    output_queue = Queue()

    num_workers = 8
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


def test(args):
    print('*'*80)
    print('*'*10 + '  TEST  ' + '*'*10)
    if args.debug == True:
        pdb.set_trace()

    num_samples_per_img = args.num_samples_per_img
    input_dir = args.input_dir

    cv2.namedWindow('image_a_b', 0)
    cv2.namedWindow('patch_a_b', 0)
    for i in glob.iglob(os.path.join(input_dir, '*.jpg')):
        # Read as grayscale
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

        target_size = (320, 240) # (w, h)
        patch_size = (128, 128)

        if img.shape[:2][::-1] < target_size:
            continue

        for _ in range(num_samples_per_img):
            img_a = scale_down(img, target_size)
            img_a = center_crop(img_a, target_size)
            assert img_a.shape[::-1] == target_size
            
            pts_orig, pts_perturbed = generate_points()
            patch_a = crop(img_a, pts_orig[0], patch_size)
            img_b = warp(img_a, pts_perturbed, pts_orig, target_size)
            patch_b = crop(img_b, pts_orig[0], patch_size)

            img_a_show = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
            for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                img_a_show = cv2.line(img_a_show, (pts_orig[k1][0], pts_orig[k1][1]), 
                        (pts_orig[k2][0], pts_orig[k2][1]), (255, 0, 0), 2)
                img_a_show = cv2.line(img_a_show, (pts_perturbed[k1][0], pts_perturbed[k1][1]), 
                        (pts_perturbed[k2][0], pts_perturbed[k2][1]), (0, 255, 0), 2)
            img_b_show = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
            for (k1, k2) in [[0, 1], [1, 3], [3, 2], [2, 0]]:
                img_b_show = cv2.line(img_b_show, (pts_orig[k1][0], pts_orig[k1][1]), 
                        (pts_orig[k2][0], pts_orig[k2][1]), (255, 0, 0), 2)
            cv2.imshow('image_a_b', np.concatenate((img_a_show, img_b_show), 1))
            cv2.imshow('patch_a_b', np.concatenate((patch_a, patch_b), 1))

            offset = (pts_perturbed - pts_orig).reshape(-1)
            assert len(offset) == 8

            key = cv2.waitKey(0)
            if key in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_samples_per_img', type=int)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    if args.test == True:
        test(args)
    else:
        main(args)