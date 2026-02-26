import tensorflow as tf
import numpy as np


# 3 ops that have parameters
op_list = ['expand', 'depthwise', 'project']


# 分别计算L1范数
def select_elastic_channel_both(layers, index, channel, random=False):
    layer = layers[index]
    weights = layer.get_weights()
    # compute the l1 norm
    start = layer.name.find('_', 7)
    name = layer.name[start + 1:]
    selected_weights, selected_bn = [], []
    if name == op_list[0]:
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 3))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_weights = tf.gather(weights, axis=4, indices=selected_idx)
            # re_organize BN layer
            bn_weights = layers[index + 1].get_weights()
            selected_bn = tf.gather(bn_weights, axis=1, indices=selected_idx)
        else:
            selected_weights = weights
            selected_bn = layers[index + 1].get_weights()
    elif name == op_list[1]:
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 4))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_weights = tf.gather(weights, axis=3, indices=selected_idx)
            bn_weights = layers[index + 1].get_weights()
            selected_bn = tf.gather(bn_weights, axis=1, indices=selected_idx)
        else:
            selected_weights = weights
            selected_bn = layers[index + 1].get_weights()
    else:
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 4))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_weights = tf.gather(weights, axis=3, indices=selected_idx)
        else:
            selected_weights = weights
        selected_bn = layers[index + 1].get_weights()

    return selected_weights, selected_bn


# 根据PW_expand卷积通道排序结果选择DW和PW_project channels
def select_elastic_channel_pw(layers, index, channel, expand_ratio=6, random=False):
    if expand_ratio == 1:
        index = index + 2

    layer = layers[index]
    weights = layer.get_weights()
    start = layer.name.find('_', 7)
    name = layer.name[start + 1:]

    selected_params = []
    # 判断block是否需要expand，没有expand的block只有4个layer
    if expand_ratio == 1:
        assert name == op_list[1]
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 4))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_params.append(tf.gather(weights, axis=3, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+1].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+2].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(layers[index+3].get_weights())
        else:
            for i in range(4):
                selected_params.append(layers[index + i].get_weights())
    # expand_block有6个layer
    else:
        assert name == op_list[0]
        # sort and select
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 3))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_params.append(tf.gather(weights, axis=4, indices=selected_idx))
            # pw_expand卷积后的dw和pw卷积选择通道
            selected_params.append(tf.gather(layers[index+1].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+2].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+3].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+4].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(layers[index+5].get_weights())
        else:
            for i in range(6):
                selected_params.append(layers[index + i].get_weights())

    return selected_params


def select_elastic_channel_dw(layers, index, channel, expand_ratio=6, random=False):
    if expand_ratio == 1:
        index = index + 2

    selected_params = []
    if expand_ratio == 1:
        weights = layers[index].get_weights()
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 4))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_params.append(tf.gather(weights, axis=3, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+1].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+2].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(layers[index+3].get_weights())
        else:
            for i in range(4):
                selected_params.append(layers[index + i].get_weights())
    else:
        weights = layers[index+2].get_weights()   # dw_conv weights
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 4))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_params.append(tf.gather(layers[index].get_weights(), axis=4, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+1].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(weights, axis=3, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+3].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+4].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(layers[index+5].get_weights())
        else:
            for i in range(6):
                selected_params.append(layers[index + i].get_weights())

    return selected_params


# new selecting function
def select_elastic_channel_new(layers, index, channel, expand_ratio=6, random=False):
    if expand_ratio == 1:
        index += 2

    layer = layers[index]
    weights = layer.get_weights()
    start = layer.name.find('_', 7)
    name = layer.name[start + 1:]
    selected_params = []
    if expand_ratio == 1:
        assert name == op_list[1]
        importance = tf.reduce_sum(tf.abs(weights), axis=(0, 1, 2, 4))
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_params.append(tf.gather(weights, axis=3, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+1].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+2].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(layers[index+3].get_weights())
        else:
            for i in range(4):
                selected_params.append(layers[index + i].get_weights())
        # expand_ratio > 1
    else:
        assert name == op_list[0]
        # sort and select
        normalize_pw = tf.reduce_sum(tf.abs(weights), axis=(0,1,2,3)) / tf.reduce_sum(tf.abs(weights))
        dw_w = layers[index+2].get_weights()
        normalize_dw = tf.reduce_sum(tf.abs(dw_w), axis=(0,1,2,4)) / tf.reduce_sum(tf.abs(dw_w))
        importance = tf.multiply(normalize_pw, normalize_dw)
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
        if random == True:
            sorted_idx = tf.random.shuffle(sorted_idx)
        if channel < len(sorted_idx):
            selected_idx = sorted_idx[:channel]
            selected_params.append(tf.gather(weights, axis=4, indices=selected_idx))
            # pw_expand卷积后的dw和pw卷积选择通道
            selected_params.append(tf.gather(layers[index+1].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+2].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+3].get_weights(), axis=1, indices=selected_idx))
            selected_params.append(tf.gather(layers[index+4].get_weights(), axis=3, indices=selected_idx))
            selected_params.append(layers[index+5].get_weights())
        else:
            for i in range(6):
                selected_params.append(layers[index + i].get_weights())

    return selected_params