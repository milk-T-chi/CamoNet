import tensorflow as tf
import Model
from Model import MobileNet_base
from weight_sharing import reorganize_model
from keras_flops import get_flops


def insert_residual_all(model, kernel_size=3, downsample_ratio=2, channel_reduce_ratio=1):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1, len(layers)):
        if "inverted_residual" in layers[i].name:
            in_channels = layers[i].input.shape[3]
            out_channels = layers[i].output.shape[3]
            in_size = layers[i].input.shape[2]
            out_size = layers[i].output.shape[2]

            while in_size / downsample_ratio < kernel_size:
                kernel_size -= 2

            x1 = Model.SiameseResidual(in_channels=in_channels, out_channels=out_channels,
                                       out_size=out_size, kernel_size=kernel_size,
                                       downsample_ratio=downsample_ratio,
                                       channel_reduce_ratio=channel_reduce_ratio)(x, training=True)

            x2 = layers[i](x, training=True)
            x = x1 + x2
        else:
            x = layers[i](x)

    new_model = tf.keras.Model(layers[0].input, x)
    return new_model


def zero_initializer(model):
    initializer = tf.keras.initializers.RandomUniform(minval=-1e-30, maxval=1e-30)

    for layer in model.layers:
        if layer.name.__contains__("siamese_residual"):
            for sublayer in layer.get_conv_layers():
                kernal_shape = sublayer.get_weights()[0].shape
                # bias_shape = sublayer.get_weights()[1].shape
                zero_param = [initializer(shape=kernal_shape)]
                sublayer.set_weights(zero_param)
