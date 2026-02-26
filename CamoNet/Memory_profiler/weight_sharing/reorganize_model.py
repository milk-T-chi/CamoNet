'''
    使用须知：
        （1）父结构（预训练模型）与子结构必须保持alpha相同；
        （2）搜索空间必须保持stage_1固定不变，之后的6个stage中块的个数n和倍乘因子t需要小于预训练模型setting中的大小；
        （3）select_elastic_channel_both()表示DW、PW卷积层的channel选取是不一致的，分别计算了它们的L1范数，select_elastic_channel_pw()
        和select_elastic_channel_dw()分别为按照pw和dw卷积通道排序的选取结果。
'''


from random import sample
from weight_sharing.select_channel import *

random_depth = False    # 是否选择随机深度的block
random_width = False    # 是否选择随机宽度的channel


def set_random_depth(random=False):
    random_depth = random


def set_random_width(random=False):
    random_width = random


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def reorganize_model(pretrained_model, model, inverted_residual_setting, alpha=1.0, choice='pw'):
    _pretrained_layers = pretrained_model.layers
    layers = model.layers
    pretrained_layers = []
    for layer in _pretrained_layers:
        if len(layer.get_weights()) != 0:
            pretrained_layers.append(layer)  # remove non-trainable layer

    # set the front layers
    pretrained_idx = 0
    conv_dw_layers = layers[1].get_sub_layers()
    conv_dw_layers.extend(layers[2].get_sub_layers())  # get the front layers
    for layer in conv_dw_layers:
        if pretrained_layers[pretrained_idx].name.find('block_1') == 0:  # block_1
            break
        else:
            if np.shape(layer.get_weights()) != np.shape(pretrained_layers[pretrained_idx].get_weights()):
                print('name1:', pretrained_layers[pretrained_idx].name, '<--->', 'name:', layer.name)
                return None
            layer.set_weights(pretrained_layers[pretrained_idx].get_weights())
            pretrained_idx = pretrained_idx + 1

    # set the elastic inverted_residual_block channels
    set_elastic_channel(pretrained_layers, layers, pretrained_idx, 3, inverted_residual_setting, alpha=alpha, choice=choice)

    return model


def set_elastic_channel(pretrained_layers, layers, pretrained_idx, idx, inverted_residual_setting, alpha=1.0, choice='both'):
    # 从第1个inverted_residual_block开始
    assert pretrained_layers[pretrained_idx].name.find('block_1') == 0

    start_idx = [0, 12, 30, 54, 72, 90]  # 6 stages
    blks = [[1], [1, 2], [1, 2, 3], [1, 2], [1, 2], [0]]

    if choice == 'both':
        for i in range(1, len(inverted_residual_setting)):  # skip the first fixed stage, 1~6 stage
            setting = inverted_residual_setting[i]
            pre_setting = inverted_residual_setting[i - 1]  # get last setting
            channel = setting[0] * _make_divisible(setting[1] * alpha, 8)
            in_channel = setting[0] * _make_divisible(pre_setting[1] * alpha, 8)    # t * in_channel
            numbers = setting[2]  # n
            # set the first block, because its channel equals input_channel(in_channel)
            sub_layers = layers[idx].get_sub_layers()
            idx = idx + 1
            if setting[0] == 1:
                re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + 2,
                                  sub_layers, in_channel, 'both')
            else:
                re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1],
                                  sub_layers, in_channel, 'both')
            sub_layers.clear()

            blk = sorted(sample(blks[i - 1], numbers-1))      # sample blocks in the stage_i

            # set left blocks
            for j in range(1, numbers):
                sub_layers = layers[idx].get_sub_layers()
                idx = idx + 1
                if random_depth == False:
                    if setting[0] == 1:
                        re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + j * 6 + 2,
                                          sub_layers, channel, 'both')
                    else:
                        re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + j * 6,
                                          sub_layers, channel, 'both')
                else:
                    if setting[0] == 1:
                        re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + blk[j - 1] * 6 + 2,
                                          sub_layers, channel, 'both')
                    else:
                        re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + blk[j - 1] * 6,
                                          sub_layers, channel, 'both')
    elif choice in ['pw', 'dw', 'new']:
        for i in range(1, len(inverted_residual_setting)):  # skip the first fixed stage, 1~6 stage
            setting = inverted_residual_setting[i]
            pre_setting = inverted_residual_setting[i - 1]  # get last setting
            channel = setting[0] * _make_divisible(setting[1] * alpha, 8)
            in_channel = setting[0] * _make_divisible(pre_setting[1] * alpha, 8)    # t * in_channel
            numbers = setting[2]  # n
            # set the first block, because its channel equals input_channel(in_channel)
            sub_layers = layers[idx].get_sub_layers()
            idx = idx + 1
            re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1],
                              sub_layers, in_channel, 'pw', expand_ratio=setting[0])
            sub_layers.clear()

            blk = sorted(sample(blks[i - 1], numbers-1))  # sample blocks in the stage_i

            for j in range(1, numbers):
                sub_layers = layers[idx].get_sub_layers()
                idx = idx + 1
                if random_depth == False:
                    re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + j * 6,
                                      sub_layers, channel, choice=choice, expand_ratio=setting[0])
                else:
                    re_organize_block(pretrained_layers, pretrained_idx + start_idx[i - 1] + blk[j - 1] * 6,
                                      sub_layers, channel, choice=choice, expand_ratio=setting[0])


# 根据channel数重组block
def re_organize_block(pretrained_layers, idx, sub_layers, _channel, choice, expand_ratio=6):
    if choice == 'both':
        for i in range(len(sub_layers)):
            loc = pretrained_layers[idx + i].name.find('_', 7)
            name = pretrained_layers[idx + i].name[loc + 1:]
            if name in op_list:
                weight, bn_weight = select_elastic_channel_both(pretrained_layers, idx + i, _channel, random=random_width)
                if np.shape(weight) != np.shape(sub_layers[i].get_weights()):
                    print('name:', pretrained_layers[idx+i].name, '->', sub_layers[i].name)
                sub_layers[i].set_weights(weight)
                sub_layers[i + 1].set_weights(bn_weight)
    elif choice == 'pw':
        weight_list = select_elastic_channel_pw(pretrained_layers, idx, _channel, expand_ratio=expand_ratio, random=random_width)
        for i in range(len(sub_layers)):
            sub_layers[i].set_weights(weight_list[i])
    elif choice == 'dw':
        weight_list = select_elastic_channel_dw(pretrained_layers, idx, _channel, expand_ratio=expand_ratio, random=random_width)
        for i in range(len(sub_layers)):
            sub_layers[i].set_weights(weight_list[i])
    elif choice == 'new':
        weight_list = select_elastic_channel_new(pretrained_layers, idx, _channel, expand_ratio=expand_ratio, random=random_width)
        for i in range(len(sub_layers)):
            sub_layers[i].set_weights(weight_list[i])
    else:
        raise ValueError("choice is wrong")
