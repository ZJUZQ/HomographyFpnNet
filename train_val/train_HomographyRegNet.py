#coding=utf-8
"""
Example usage:

"""
from __future__ import absolute_import
import os, sys
import _init_path

import argparse
import math
import glob
import pdb
import tensorflow as tf
import json
import copy

from homographynet import data
from homographynet.callbacks import LearningRateScheduler, LearningRateScheduler_CosineDecay
from homographynet.losses import mean_corner_error_4pts
from homographynet.model import create_homographynet_vgg, create_homographynet_vgg_fpn
from homographynet.model import create_homographynet_mobilenet_v2

MODEL_MAP = {'vgg': create_homographynet_vgg,
            'vgg_fpn': create_homographynet_vgg_fpn,
            'mobilenet_v2': create_homographynet_mobilenet_v2,
            }
mean_corner_error_map = {4: mean_corner_error_4pts,
                    }

parser = argparse.ArgumentParser()
parser.add_argument('train_data_dir', type=str)
parser.add_argument('val_data_dir', type=str)
parser.add_argument('checkpoint_dir', type=str)
parser.add_argument('--finetune_weight_file', type=str)

parser.add_argument('--input_height', type=int, default=128)
parser.add_argument('--input_width', type=int, default=128)

parser.add_argument('--num_pts', type=int, default=4)
parser.add_argument('--model_name', type=str, default='mobilenet_v2')

parser.add_argument('--pooling', type=str, default='avg')
parser.add_argument('--alpha', type=float, default=1.0, help='channel number multiplier')

# Configuration from paper
parser.add_argument('--lr_scheduler', type=str, default='cosine_decay')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_steps', type=int, default=90000)
parser.add_argument('--init_lr', type=float, default=0.005)

# for step_decay
parser.add_argument('--step_size', type=int, default=30000)
parser.add_argument('--decay_rate', type=float, default=0.1)

# for cosine_decay
parser.add_argument('--warm_steps', type=int, default=1000)

parser.add_argument('--grad_clip', type=float)
parser.add_argument('--random_light', action='store_true', default=False)
parser.add_argument('--random_erase', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()


_SAMPLES_PER_ARCHIVE = 6400*2 # set in dataset/generate.py
TRAIN_SAMPLES = len(glob.glob(os.path.join(args.train_data_dir, '*.npz'))) * _SAMPLES_PER_ARCHIVE
VAL_SAMPLES = len(glob.glob(os.path.join(args.val_data_dir, '*.npz'))) * _SAMPLES_PER_ARCHIVE


def main():
    if args.debug == True:
        pdb.set_trace()

    assert args.model_name in MODEL_MAP
    input_shape = (args.input_height, args.input_width, 2)
    model = MODEL_MAP[args.model_name](input_shape=input_shape, num_pts=args.num_pts, pooling=args.pooling,
                alpha=args.alpha)
    
    if args.finetune_weight_file is not None:
        #pretrain_model = tf.keras.models.load_model(args.finetune_weight_file, compile=False)
        pretrain_model = MODEL_MAP[args.model_name](weights_file=args.finetune_weight_file, input_shape=input_shape, 
            pooling=args.pooling, alpha=args.alpha)
        #param_dict = {}
        for layer in pretrain_model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                #param_dict[layer.name] = copy.deepcopy(weights)
                model.get_layer(layer.name).set_weights(weights)
        del pretrain_model
    else:
        pass

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # save model.png
    tf.keras.utils.plot_model(model, to_file=os.path.join(args.checkpoint_dir, 'model.png'),
                show_shapes=True)

    batch_size = args.batch_size
    target_iterations = args.max_steps # in paper 90000 for batch_size = 64
    base_lr = args.init_lr

    if args.grad_clip is not None:
        sgd = tf.keras.optimizers.SGD(lr=base_lr, momentum=0.9, clipnorm=args.grad_clip)
    else:
        sgd = tf.keras.optimizers.SGD(lr=base_lr, momentum=0.9)
    
    # note, for 'vgg_siamese_multistage', single loss and metric will apply to each output
    #   refer to https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
    model.compile(optimizer=sgd, loss=['mean_squared_error'], metrics=[mean_corner_error_map[args.num_pts]])
    model.summary()
        
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.checkpoint_dir, 'model.{epoch:02d}.h5'),
                monitor='val_loss', save_best_only=True)
    
    # In the paper, the 90,000 iterations was for batch_size = 64
    # So scale appropriately
    #target_iterations = int(target_iterations * 64 / batch_size)

    # LR scaling as described in the paper
    if args.lr_scheduler == 'step_decay':
        lr_scheduler = LearningRateScheduler(base_lr, args.decay_rate, args.step_size)
    elif args.lr_scheduler == 'cosine_decay':
        lr_scheduler = LearningRateScheduler_CosineDecay(base_lr, args.warm_steps, target_iterations)

    # As stated in Keras docs
    steps_per_epoch = int(TRAIN_SAMPLES / batch_size)
    epochs = int(math.ceil(target_iterations / steps_per_epoch))

    loader = data.loader_images_pts(args.train_data_dir, batch_size, 
        b_random_light=args.random_light, b_random_erase=args.random_erase)
    val_loader = data.loader_images_pts(args.val_data_dir, batch_size)
    val_steps = int(VAL_SAMPLES / batch_size)

    # Train
    hist = model.fit_generator(loader, steps_per_epoch, epochs,
                        callbacks=[lr_scheduler, checkpoint],
                        validation_data=val_loader, validation_steps=val_steps)
    
    with open(os.path.join(args.checkpoint_dir, 'hist.json'), 'w') as f:
        json.dump(hist.history, f)


if __name__ == '__main__':
    main()