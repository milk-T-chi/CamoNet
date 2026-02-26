import tensorflow as tf
import numpy as np


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SiameseResidual(tf.keras.Model):
    def __init__(self, in_channels, out_channels, out_size,
                 kernel_size=5,
                 downsample_ratio=2, upsample_type='bilinear',
                 channel_reduce_ratio=1, stride=1, **kwargs):
        super(SiameseResidual, self).__init__(**kwargs)
        layer_list = []

        layer_list.extend([
            tf.keras.layers.AveragePooling2D(pool_size=(downsample_ratio, downsample_ratio)),
            tf.keras.layers.Conv2D(filters=int(in_channels / channel_reduce_ratio),
                                   kernel_size=kernel_size,
                                   strides=(stride, stride),
                                   # groups=n_groups,
                                   padding="same",
                                   use_bias=False,
                                   # kernel_regularizer=tf.keras.regularizers.L2(1e-3)
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(max_value=6.0),
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=(stride, stride),
                                   # groups=n_groups,
                                   padding="same",
                                   use_bias=False,
                                   # kernel_regularizer=tf.keras.regularizers.L2(1e-3)
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Resizing(height=out_size, width=out_size,
                                     interpolation=upsample_type),
        ])
        self.main_branch = tf.keras.Sequential(layer_list)

    def get_sub_layers(self):
        layers = self.main_branch.layers
        for layer in layers:
            if len(layer.get_weights()) == 0:
                layers.remove(layer)
        return layers

    def call(self, inputs, training=True, **kwargs):
        return self.main_branch(inputs, training=training)

    def get_conv_layers(self):
        return [self.main_branch.layers[1], self.main_branch.layers[4]]


class ConvBNReLU(tf.keras.Model):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)

        layer_list = []
        layer_list.extend([
            # 3x3 depthwise conv
            tf.keras.layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                   strides=stride, padding='SAME', use_bias=False, name='Conv2d'),
            tf.keras.layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name='BatchNorm'),
            tf.keras.layers.ReLU(max_value=6.0),
        ])
        self.main_branch = tf.keras.Sequential(layer_list)

    def call(self, inputs, training=True, **kwargs):
        return self.main_branch(inputs, training=training)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv': self.conv,
            'bn': self.bn,
            'activation': self.activation,
        })
        return config

    def get_sub_layers(self):
        layers = self.main_branch.layers
        for layer in layers:
            if len(layer.get_weights()) == 0:
                layers.remove(layer)
        return layers


class InvertedResidual(tf.keras.Model):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer_list = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))

        layer_list.extend([
            # 3x3 depthwise conv
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                            use_bias=False, name='depthwise'),
            tf.keras.layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name='depthwise/BatchNorm'),
            tf.keras.layers.ReLU(max_value=6.0),
            # 1x1 pointwise conv(linear)
            tf.keras.layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                                   padding='SAME', use_bias=False, name='project'),
            tf.keras.layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name='project/BatchNorm')
        ])
        self.main_branch = tf.keras.Sequential(layer_list, name='expanded_conv')

    def call(self, inputs, training=True, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs, training=training)
        else:
            return self.main_branch(inputs, training=training)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_channel': self.hidden_channel,
            'use_shortcut': self.use_shortcut,
            'main_branch': self.main_branch,
        })
        return config

    def get_sub_layers(self):
        if self.main_branch.layers[0].name == 'expand':
            layers = self.main_branch.layers[0].get_sub_layers()
            layers.extend(self.main_branch.layers[1:])
        else:
            layers = self.main_branch.layers
        for layer in layers:
            if len(layer.get_weights()) == 0:
                layers.remove(layer)
        return layers

    def get_all_layers(self):
        if self.main_branch.layers[0].name == 'expand':
            layer_list = self.main_branch.layers[0].get_all_layers()
            layer_list.extend(self.main_branch.layers[1:])
        else:
            layer_list = self.main_branch.layers
        return layer_list


def MobileNet_base(im_height=224,
                   im_width=224,
                   num_classes=5,
                   alpha=1,
                   round_nearest=8,
                   include_top=True,
                   inverted_residual_setting=None):
    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
    block = InvertedResidual
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)

    input_image = tf.keras.layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    # conv1
    x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
    # building inverted residual blocks
    for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            x = block(x.shape[-1],
                      output_channel,
                      stride,
                      expand_ratio=t)(x)
    # building last several layers
    x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)

    if include_top is True:
        # building classifier
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  # pool + flatten
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(num_classes, name='Logits')(x)
    else:
        output = x

    model = tf.keras.Model(inputs=input_image, outputs=output)
    return model, inverted_residual_setting, alpha
