import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Activation
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import tensorflow.keras.layers as tfkl
from keras.utils.generic_utils import get_custom_objects
from Model import *
from keras_flops import get_flops
# from shufflenet import ChannelShuffle,ConvBNReLU
import os
from LiteSiamese import get_encoder
from residual_utils import insert_residual_all, zero_initializer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def swish(inputs):
    return (K.sigmoid(inputs) * inputs)


def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6


__all__ = ['count_param_size', 'count_activation_size', 'profile_memory_cost', 'get_flops']


def get_flops_v2(model, batch_size=None):
    if batch_size is None:
        batch_size = 1
    flops = get_flops(model, batch_size)
    return flops


def count_param_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
    frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits
    trainable_param_size = 0
    frozen_param_size = 0
    for layer in net.layers:
        # print(layer.name)
        trainable_var_list = layer.trainable_weights
        non_trainable_var_list = layer.non_trainable_weights
        train_layer_weigtht_count = sum([np.prod(w.shape) for w in trainable_var_list])
        trainable_param_size += trainable_param_bits / 8 * train_layer_weigtht_count
        non_train_layer_weigtht_count = sum([np.prod(w.shape) for w in non_trainable_var_list])
        frozen_param_size += frozen_param_bits / 8 * non_train_layer_weigtht_count
        # print(train_layer_weigtht_count, non_train_layer_weigtht_count)
    model_size = trainable_param_size + frozen_param_size
    # if print_log:
    #     print('Total: %d' % model_size,
    #           '\tTrainable: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
    #           '\tFrozen: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
    # Byte
    return model_size


def Conv_act(layer, act_byte):
    if layer.trainable:
        grad_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    else:
        grad_activations_now = 0
    tmp_activations_now = np.prod(layer.input_shape[1:]) * act_byte + np.prod(
        layer.output_shape[1:]) * act_byte // layer.groups
    return grad_activations_now, tmp_activations_now


def Dense_act(layer, act_byte):
    if layer.trainable:
        grad_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    else:
        grad_activations_now = 0
    tmp_activations_now = np.prod(layer.input_shape[1:]) * act_byte + np.prod(
        layer.output_shape[1:]) * act_byte
    return grad_activations_now, tmp_activations_now


def BN_act(layer, act_byte):
    if layer.trainable:
        grad_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    else:
        grad_activations_now = 0
    tmp_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    return grad_activations_now, tmp_activations_now


def RELU_act(layer, act_byte):
    if layer.trainable:
        grad_activations_now = np.prod(layer.input_shape[1:]) / 8
    else:
        grad_activations_now = 0
    tmp_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    return grad_activations_now, tmp_activations_now


def Activation_act(layer, act_byte):
    if layer.trainable:
        grad_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    else:
        grad_activations_now = 0
    tmp_activations_now = np.prod(layer.input_shape[1:]) * act_byte
    return grad_activations_now, tmp_activations_now


def count_activation_size_layer(layer, act_byte):
    # 函数绑定，将激活函数swish和h_swish添加到keras框的Activation类中
    get_custom_objects().update({'swish': Activation(swish)})
    get_custom_objects().update({'h_swish': Activation(h_swish)})
    if isinstance(layer, tfkl.Dense):
        return Dense_act(layer, act_byte)

    elif type(layer) in [tfkl.Conv1D, tfkl.Conv3D, tfkl.Conv2D, tfkl.DepthwiseConv2D]:
        return Conv_act(layer, act_byte)

    elif isinstance(layer, tfkl.BatchNormalization):
        return BN_act(layer, act_byte)

    elif type(layer) in [tfkl.LeakyReLU, tfkl.PReLU, tfkl.ReLU, tfkl.ThresholdedReLU]:
        return RELU_act(layer, act_byte)

    elif type(layer) in [tfkl.Softmax, Activation]:
        return Activation_act(layer, act_byte)

    else:
        return 0, 0


def count_activation_size(baseline_model, require_backward=True, activation_bits=32):
    act_byte = activation_bits / 8
    base_layer_list = [tfkl.Dense, tfkl.Conv1D, tfkl.Conv2D, tfkl.Conv3D, tfkl.LeakyReLU, tfkl.PReLU, tfkl.ReLU,
                       tfkl.ThresholdedReLU, tfkl.Softmax, tfkl.DepthwiseConv2D, tfkl.BatchNormalization]
    baseline_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    memory_info_dict = {
        'grad_activation_size': tf.convert_to_tensor(np.zeros(1)),
        'max_temp_activation_size': tf.convert_to_tensor(np.zeros(1)),
    }
    for layer in baseline_model.layers:
        if layer.submodules == ():
            grad_activations_now, tmp_activations_now = count_activation_size_layer(layer, act_byte)
            current_activation = memory_info_dict['grad_activation_size'] + tmp_activations_now
            memory_info_dict['max_temp_activation_size'] = max(
                current_activation, memory_info_dict['max_temp_activation_size']
            )
            memory_info_dict['grad_activation_size'] += grad_activations_now
        else:
            # print(layer.name)
            for sub_layer in layer.get_sub_layers():
                # print(sub_layer)
                grad_activations_now, tmp_activations_now = count_activation_size_layer(sub_layer, act_byte)
                current_activation = memory_info_dict['grad_activation_size'] + tmp_activations_now
                memory_info_dict['max_temp_activation_size'] = max(
                    current_activation, memory_info_dict['max_temp_activation_size']
                )
                memory_info_dict['grad_activation_size'] += grad_activations_now

    return memory_info_dict['max_temp_activation_size'].numpy()[0], memory_info_dict['grad_activation_size'].numpy()[0]


def profile_memory_cost(baseline_model, require_backward=True,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=8):
    param_size = count_param_size(baseline_model, trainable_param_bits, frozen_param_bits, print_log=True)
    temp_activation_size, grad_activation_size = count_activation_size(baseline_model, require_backward, activation_bits)

    memory_cost = temp_activation_size * batch_size + param_size
    param_size_MB = param_size / (1024 * 1024)
    temp_activation_size_MB = temp_activation_size / (1024 * 1024)
    memory_cost_MB = memory_cost / (1024 * 1024)
    grad_activation_size = grad_activation_size / (1024 * 1024)
    return memory_cost_MB, {'param_size': param_size_MB, 'act_size': temp_activation_size_MB,
                            'grad_activation_size': grad_activation_size}
                            
    # return memory_cost, {'param_size': param_size, 'act_size': activation_size}


if __name__ == '__main__':
    encoder = get_encoder(
        trainable=True,
        weights="imagenet",
        image_size=224,
        image_channels=3,
        alpha=1.4)

    # encoder = insert_residual_all(model=encoder, kernel_size=3,
    #                               channel_reduce_ratio=1,
    #                               downsample_ratio=2)

    # zero_initializer(encoder)
    # encoder = tf.keras.applications.ResNet50(
    #     include_top=True,
    #     weights="imagenet",
    #     input_tensor=None,
    #     input_shape=(224, 224, 3),
    #     pooling=None,
    # )
    # encoder.summary()

    memory_info = profile_memory_cost(encoder, batch_size=16)
    print(memory_info)
