#coding=utf-8
import os
import glob
import numpy as np

def _shuffle_in_unison(a, b):
    """A hack to shuffle both a and b the same "random" way"""
    prng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(prng_state)
    np.random.shuffle(b)

def loader_homographynet_images_offsets(path, batch_size=64, normalize=True, num_stage=1):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            del archive
            _shuffle_in_unison(images, offsets)
            # Split into mini batches
            num_batches = int(len(offsets) / batch_size)
            images = np.array_split(images, num_batches)
            offsets = np.array_split(offsets, num_batches)
            while offsets:
                batch_images = images.pop()
                batch_offsets = offsets.pop()
                if normalize:
                    batch_images = (batch_images - 127.5) / 127.5
                    batch_offsets = batch_offsets / 32. # perturb [-32, 32]
                    #batch_offsets = batch_offsets / 128.
                yield batch_images, [batch_offsets] * num_stage


def loader_images_pts(path, batch_size=64, normalize=True, num_stage=1, b_random_light=False, b_random_erase=False):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            pts = archive['pts'] # normalized pts in image_orig
            del archive
            _shuffle_in_unison(images, pts)
            # Split into mini batches
            num_batches = int(len(pts) / batch_size)
            images = np.array_split(images, num_batches)
            pts = np.array_split(pts, num_batches)
            while pts:
                batch_images = images.pop()
                batch_pts = pts.pop()
                #batch_pts = batch_pts.reshape(batch_pts.shape[0], 8)
                if b_random_light == True:
                    # adjust brightness to image_a
                    for i in range(batch_images.shape[0]):
                        batch_images[i, :, :, 0] = augment_brightness_grayscale(batch_images[i, :, :, 0])

                if normalize:
                    batch_images = (batch_images - 127.5) / 127.5 # [-1, 1]
                
                if b_random_erase == True:
                    # 在image_warp中随机改变一小块区域的信息（模拟爬壁机器人在墙面上的运动）
                    for i in range(batch_images.shape[0]):
                        if np.random.randint(0, 2) == 0:
                            h, w = np.random.randint(4, 16, 2, dtype=np.int32)
                            x = np.random.randint(2, 128-w-2, dtype=np.int32)
                            y = np.random.randint(2, 128-h-2, dtype=np.int32)
                            batch_images[i, y:y+h, x:x+w, 1] = np.random.rand(h, w)*2. - 1. # [-1, 1]

                yield batch_images, [batch_pts] * num_stage
        
def augment_brightness_grayscale(gray_image):
    gray_image = np.array(gray_image, dtype = np.float64)
    random_bright = np.random.uniform(low=0.5, high=1.5)
    gray_image = gray_image * random_bright
    gray_image[gray_image[:, :] > 255]  = 255
    gray_image = np.array(gray_image, dtype = np.uint8)
    return gray_image