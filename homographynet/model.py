#coding=utf-8
import os.path

import tensorflow as tf
from keras_applications import mobilenet_v2
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, Conv2DTranspose, \
    BatchNormalization, Flatten, Concatenate, GlobalMaxPool2D, Input, Add, GlobalAveragePooling2D, UpSampling2D


#def create_model(weights_file=None):
def create_homographynet_vgg(weights_file=None, input_shape=(128, 128, 2), num_pts=4, pooling=None, **kwargs):
    """
    Use fully convolutional structure to support any input shape
    """
    model = Sequential(name='homographynet')
    model.add(InputLayer(input_shape, name='input_1'))
    #model.add(InputLayer((None, None, 2), name='input_1')) # support any input shape

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128]
    for i, f in enumerate(filters, start=1):
        model.add(Conv2D(filters=f, kernel_size=3, padding='same', 
                    activation='relu', name='conv2d_{}'.format(i)))
        model.add(BatchNormalization(axis=-1, name='batch_normalization_{}'.format(i)))
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != 8:
            model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2), 
                    name='max_pooling2d_{}'.format(int(i/2))))
    if pooling == 'max':
        model.add(GlobalMaxPool2D(name='global_max_pool2d')) # to support any input shape
    elif pooling == 'avg':
        model.add(GlobalAveragePooling2D(name='global_avg_pool2d')) # to support any input shape
    elif pooling == 'flatten':
        model.add(Flatten(name='flatten_1'))
        model.add(Dropout(rate=0.5, name='dropout_1'))
    else:
        raise ValueError

    model.add(Dense(units=1024, activation='relu', name='dense_1'))
    model.add(Dropout(rate=0.5, name='dropout_2'))

    # Regression model
    model.add(Dense(units=2*num_pts, name='dense_2')) # no activation

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


def create_homographynet_vgg_fpn(weights_file=None, input_shape=(128, 128, 2), num_pts=4, **kwargs):
    """
    Use fully convolutional structure to support any input shape
    """
    input_tensor = Input(input_shape, name='input')
    x = input_tensor

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128] + 4 * [256]
    features = []
    for i, f in enumerate(filters, start=1):
        x = Conv2D(filters=f, kernel_size=3, padding='same', 
                    activation='relu', name='conv2d_{}'.format(i))(x)
        x = BatchNormalization(axis=-1, name='batch_normalization_{}'.format(i))(x)
        if i % 2 == 0:
            features.append(x)
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != len(filters):
            x = MaxPooling2D(pool_size=(2,2), strides=(2, 2), 
                    name='max_pooling2d_{}'.format(int(i/2)))(x)
    
    P5 = Conv2D(256, (1, 1), name='fpn_c5p5')(features[-1]) # shape (4, 4)
    # 上采样之后的P4和卷积之后的 C4像素相加得到 P4
    P4 = Add(name='fpn_p4add')([
        Conv2D(256, (1, 1), name='fpn_c4p4')(features[-2]), 
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5)
        ]) # shape (8, 8)
    P3 = Add(name='fpn_p3add')([
        Conv2D(256, (1, 1), name='fpn_c3p3')(features[-3]), 
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4)
        ]) # shape (16, 16)

    # P2-P5进行一次3*3的卷积，作用是消除上采样带来的混叠效应
    P3 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)

    x = Concatenate(axis=-1)([
        GlobalAveragePooling2D(name='g_avg_p3')(P3),
        GlobalAveragePooling2D(name='g_avg_p4')(P4),
        GlobalAveragePooling2D(name='g_avg_p5')(P5),
    ])

    x = Dense(units=1024, activation='relu', name='dense_1')(x)
    x = Dropout(rate=0.5, name='dropout_2')(x)
    # Regression model
    x = Dense(units=2*num_pts, name='dense_2')(x) # no activation

    model = Model(inputs=input_tensor, outputs=x, name='homographynet_heatmap_vgg_fpn')

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


def create_homographynet_mobilenet_v2(weights_file=None, input_shape=(128, 128, 2), num_pts=4, alpha=1.0, 
        pooling=None, **kwargs):
    """
    Use fully convolutional structure to support any input shape
    """
    base_model = mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights=None, alpha=alpha)
    # The output shape just before the pooling and dense layers is: (4, 4, 1024)
    x = base_model.output

    if pooling == 'max':
        x = GlobalMaxPool2D(name='global_max_pool2d')(x) # to support any input shape
    elif pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool2d')(x) # to support any input shape
    else:
        x = Flatten(name='flatten')(x)
        x = Dropout(rate=0.5, name='dropout')(x)
    
    x = Dense(2*num_pts, name='preds')(x)

    model = Model(inputs=base_model.input, outputs=x, name='homographynet_mobilenet_v2')

    if weights_file is not None:
        model.load_weights(weights_file)
    return model


def create_homographynet_heatmap_vgg_siamese(weights_file=None, input_shape=(128, 128, 2), num_stage=1):
    """
    Use fully convolutional structure to support any input shape
    """
    # instantiate a Keras tensor
    input_tensor = Input(input_shape, name='input')

    # two gray images concatenated 2-channel input
    assert input_tensor.shape[3] == 2
    def split(input_tensor):
        input_1 = input_tensor[:, :, :, :1]
        input_2 = input_tensor[:, :, :, 1:]
        return [input_1, input_2]
    input_1, input_2 = tf.keras.layers.Lambda(split)(input_tensor)

    input_image = Input(shape=input_shape[:2] + (input_shape[2]//2,))
    x = input_image
    for i, num_filters in enumerate([32, 32, 32, 32, 32, 32], start=1):
        if i % 2 == 0:
            x = Conv2D(num_filters, 3, strides=(2, 2), padding='same', activation='relu', name='conv2d_{}'.format(i))(x)
        else:
            x = Conv2D(num_filters, 3, strides=(1, 1), padding='same', activation='relu', name='conv2d_{}'.format(i))(x)
        x = BatchNormalization(axis=-1, name='batch_norm_{}'.format(i))(x)
        #if i % 2 == 0:
        #    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_{}'.format(i//2))(x)
    model_branch = Model(inputs=input_image, outputs=x)

    part_1 = model_branch(input_1)
    part_2 = model_branch(input_2)
    bottom_feature = Concatenate(axis=-1)([part_1, part_2])

    outputs = []
    outputs_feats = []
    for s in range(1, num_stage+1):
        base_name = 'stage_{}'.format(s)
        x = bottom_feature
        if len(outputs_feats) > 0:
            x = Concatenate(name=base_name + '_concatenate', axis=-1)([x, outputs_feats[-1]])

        for i, num_filters in enumerate([64, 64, 64, 64], start=1):
            x = Conv2D(num_filters, 3, padding='same', activation='relu', name=base_name + '_conv2d_{}'.format(i))(x)
            x = BatchNormalization(axis=-1, name=base_name + '_batch_norm_{}'.format(i))(x)
        
        x = Conv2D(4, 3, padding='same', name=base_name + '_map')(x)
        outputs_feats.append(x)

        x = Conv2DTranspose(4, 3, (2, 2), padding='same', name=base_name+'_deconv1')(x)
        x = Conv2DTranspose(4, 3, (2, 2), padding='same', name=base_name+'_deconv2')(x)
        x = Conv2DTranspose(4, 3, (2, 2), padding='same', name=base_name+'_deconv3')(x)
        outputs.append(x)

    model = Model(inputs=input_tensor, outputs=outputs, name='homographynet_heatmap_vgg_siamese')

    if weights_file is not None:
        model.load_weights(weights_file)
    return model