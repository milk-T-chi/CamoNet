import tensorflow as tf
import os
from residual_utils import *


def get_mnas_backbone(shape=(224, 224, 3), residual=False):
    # load mnas model
    mnas_model = tf.keras.models.load_model("MnasNet.h5")  # modify the path

    # create new model
    input = tf.keras.layers.Input(shape=shape)
    x = input

    for i in range(1, 14):
        x = mnas_model.layers[i](x, training=True)
    
    # have residual layer
    x1 = x
    for i in range(14, 20):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r1 = insert_residual(x1, x)
        x = mnas_model.layers[20]([x, x1, r1], training=True)   # LAYER_15
    else:
        x = mnas_model.layers[20]([x, x1], training=True)

    # have residual layer
    x2 = x
    for i in range(21, 27):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r2 = insert_residual(x2, x)
        x = mnas_model.layers[27]([x, x2, r2], training=True)   # LAYER_21
    else:
        x = mnas_model.layers[27]([x, x2], training=True)

    # no residual 
    x2_1 = x
    for i in range(28, 34):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        r2_1 = insert_residual(x2_1, x)
        x = tf.keras.layers.Add()([x, r2_1])

    # have residual layer
    x3 = x
    for i in range(34, 40):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r3 = insert_residual(x3, x)
        x = mnas_model.layers[40]([x, x3, r3], training=True)   # LAYER_32
    else:
        x = mnas_model.layers[40]([x, x3], training=True)

    x4 = x
    for i in range(41, 47):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r4 = insert_residual(x4, x)
        x = mnas_model.layers[47]([x, x4, r4], training=True)   # LAYER_38
    else:
        x = mnas_model.layers[47]([x, x4], training=True)

    # no residual layer
    x4_1 = x
    for i in range(48, 54):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        r4_1 = insert_residual(x4_1, x)
        x = tf.keras.layers.Add()([x, r4_1])

    x5 = x
    for i in range(54, 60):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r5 = insert_residual(x5, x)
        x = mnas_model.layers[60]([x, x5, r5], training=True)   # LAYER_49
    else:
        x = mnas_model.layers[60]([x, x5], training=True)

    x6 = x
    for i in range(61, 67):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r6 = insert_residual(x6, x)
        x = mnas_model.layers[67]([x, x6, r6], training=True)   # LAYER_55
    else:
        x = mnas_model.layers[67]([x, x6], training=True)

    x6_1 = x
    for i in range(68, 74):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        r6_1 = insert_residual(x6_1, x)
        x = tf.keras.layers.Add()([x, r6_1])

    x7 = x
    for i in range(74, 80):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r7 = insert_residual(x7, x)
        x = mnas_model.layers[80]([x, x7, r7], training=True)   # LAYER_66
    else:
        x = mnas_model.layers[80]([x, x7], training=True)

    x7_1 = x
    for i in range(81, 87):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        r7_1 = insert_residual(x7_1, x)
        x = tf.keras.layers.Add()([x, r7_1])

    x8 = x
    for i in range(87, 93):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r8 = insert_residual(x8, x)
        x = mnas_model.layers[93]([x, x8, r8], training=True)   # LAYER_77
    else:
        x = mnas_model.layers[93]([x, x8], training=True)

    x9 = x
    for i in range(94, 100):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r9 = insert_residual(x9, x)
        x = mnas_model.layers[100]([x, x9, r9], training=True)  # LAYER_83
    else:
        x = mnas_model.layers[100]([x, x9], training=True)

    x10 = x
    for i in range(101, 107):
        x = mnas_model.layers[i](x, training=True)
    if residual:
        # add residual layer
        r10 = insert_residual(x10, x)
        x = mnas_model.layers[107]([x, x10, r10], training=True) # LAYER_89
    else:
        x = mnas_model.layers[107]([x, x10], training=True)

    for i in range(108, 116):
        x = mnas_model.layers[i](x, training=True)

    new_model = tf.keras.Model(inputs=input, outputs=x)

    return new_model


def get_ofa_backbone(shape=(224, 224, 3), residual=False):
    # load ofa model
    ofa = tf.keras.models.load_model('ofa_tx2.h5')

    # create new ofa model
    input = tf.keras.layers.Input(shape=shape)
    x = input

    for i in range(1, 14):
        x = ofa.layers[i](x, training=True)

    x1 = x
    for i in range(14, 20):
        x = ofa.layers[i](x, training=True)
    if residual:
        r1 = insert_residual(x1, x)
        x = ofa.layers[20]([x, x1, r1], training=True)   # LAYER_15
    else:
        x = ofa.layers[20]([x, x1], training=True)

    x1_1 = x
    for i in range(21, 27):
        x = ofa.layers[i](x, training=True)
    if residual:
        r1_1 = insert_residual(x1_1, x)
        x = tf.keras.layers.Add()([x, r1_1])

    x2 = x
    for i in range(27, 33):
        x = ofa.layers[i](x, training=True)
    if residual:
        r2 = insert_residual(x2, x)
        x = ofa.layers[33]([x, x2, r2], training=True)
    else:
        x = ofa.layers[33]([x, x2], training=True)

    x2_1 = x
    for i in range(34, 40):
        x = ofa.layers[i](x, training=True)
    if residual:
        r2_1 = insert_residual(x2_1, x)
        x = tf.keras.layers.Add()([x, r2_1])

    x3 = x
    for i in range(40, 46):
        x = ofa.layers[i](x, training=True)
    if residual:
        r3 = insert_residual(x3, x)
        x = ofa.layers[46]([x, x3, r3], training=True)
    else:
        x = ofa.layers[46]([x, x3], training=True)

    x3_1 = x
    for i in range(47, 53):
        x = ofa.layers[i](x, training=True)
    if residual:
        r3_1 = insert_residual(x3_1, x)
        x = tf.keras.layers.Add()([x, r3_1])

    x4 = x
    for i in range(53, 59):
        x = ofa.layers[i](x, training=True)
    if residual:
        r4 = insert_residual(x4, x)
        x = ofa.layers[59]([x, x4, r4], training=True)   # LAYER_48
    else:
        x = ofa.layers[59]([x, x4], training=True)

    x4_1 = x
    for i in range(60, 66):
        x = ofa.layers[i](x, training=True)
    if residual:
        r4_1 = insert_residual(x4_1, x)
        x = tf.keras.layers.Add()([x, r4_1])

    x5 = x
    for i in range(66, 72):
        x = ofa.layers[i](x, training=True)
    if residual:
        r5 = insert_residual(x5, x)
        x = ofa.layers[72]([x, x5, r5], training=True)
    else:
        x = ofa.layers[72]([x, x5], training=True)

    # last layers
    for i in range(73, 81):
        x = ofa.layers[i](x, training=True)
    
    new_model = tf.keras.Model(inputs=input, outputs=x)
    return new_model


def get_fbnet_backbone(shape=(224, 224, 3), residual=False):
    # load fbnet model
    model = tf.keras.models.load_model('fbnet_a_backbone.h5')

    # create new fbnet model
    input = tf.keras.layers.Input(shape=shape)
    x = input
    for i in range(1, 10):
        x = model.layers[i](x, training=True)

    x1 = x
    for i in range(10, 13):
        x = model.layers[i](x, training=True)
    x = model.layers[13](x, x1)
    if residual:
        r1 = insert_residual(x1, x)
        x = tf.keras.layers.Add()([x, r1])

    x1_1 = x
    for i in range(14, 20):
        x = model.layers[i](x, training=True)
    if residual:
        r1_1 = insert_residual(x1_1, x)
        x = tf.keras.layers.Add()([x, r1_1])

    x2 = x
    for i in range(20, 25):
        x = model.layers[i](x, training=True)
    x = model.layers[25](x, x2)
    if residual:
        r2 = insert_residual(x2, x)
        x = tf.keras.layers.Add()([x, r2])

    x3 = x
    for i in range(26, 30):
        x = model.layers[i](x, training=True)
    x = model.layers[30](x, x3)
    if residual:
        r3 = insert_residual(x3, x)
        x = tf.keras.layers.Add()([x, r3])

    x4 = x
    for i in range(31, 36):
        x = model.layers[i](x, training=True)
    x = model.layers[36](x, x4)
    if residual:
        r4 = insert_residual(x4, x)
        x = tf.keras.layers.Add()([x, r4])

    x4_1 = x
    for i in range(37, 43):
        x = model.layers[i](x, training=True)
    if residual:
        r4_1 = insert_residual(x4_1, x)
        x = tf.keras.layers.Add()([x, r4_1])

    x5 = x
    for i in range(43, 49):
        x = model.layers[i](x, training=True)
    x = model.layers[49](x, x5)
    if residual:
        r5 = insert_residual(x5, x)
        x = tf.keras.layers.Add()([x, r5])

    x6 = x
    # channel shuffle
    N,H,W,C = x.shape
    x = model.layers[50](x, (-1, H, W, C // 2))
    x = model.layers[51](x)
    x = model.layers[52](x, (-1, H, W, C))
    # dw conv
    for i in range(53, 56):
        x = model.layers[i](x, training=True)
    # group conv
    x = model.layers[56](x, 2, axis=-1)
    tmp0 = model.layers[57](x[0])
    tmp1 = model.layers[58](x[1])
    x = model.layers[59]([tmp0, tmp1], axis=3)
    x = model.layers[60](x, x6)
    if residual:
        r6 = insert_residual(x6, x)
        x = tf.keras.layers.Add()([x, r6])

    x7 = x
    for i in range(61, 67):
        x = model.layers[i](x, training=True)
    x = model.layers[67](x, x7)
    if residual:
        r7 = insert_residual(x7, x)
        x = tf.keras.layers.Add()([x, r7])

    x7_1 = x
    for i in range(68, 73):
        x = model.layers[i](x, training=True)
    if residual:
        r7_1 = insert_residual(x7_1, x)
        x = tf.keras.layers.Add()([x, r7_1])

    x8 = x
    # channel shuffle
    N,H,W,C = x.shape
    x = model.layers[73](x, (-1, H, W, C // 2))
    x = model.layers[74](x)
    x = model.layers[75](x, (-1, H, W, C))
    # dw conv
    for i in range(76, 79):
        x = model.layers[i](x, training=True)
    # group conv
    x = model.layers[79](x, 2, axis=-1)
    tmp0 = model.layers[80](x[0])
    tmp1 = model.layers[81](x[1])
    x = model.layers[82]([tmp0, tmp1], axis=3)
    x = model.layers[83](x, x8)
    if residual:
        r8 = insert_residual(x8, x)
        x = tf.keras.layers.Add()([x, r8])

    x9 = x
    for i in range(84, 90):
        x = model.layers[i](x, training=True)
    x = model.layers[90](x, x9)
    if residual:
        r9 = insert_residual(x9, x)
        x = tf.keras.layers.Add()([x, r9])

    x10 = x
    # channel shuffle
    N,H,W,C = x.shape
    x = model.layers[91](x, (-1, H, W, C // 2))
    x = model.layers[92](x)
    x = model.layers[93](x, (-1, H, W, C))
    # dw conv
    for i in range(94, 96):
        x = model.layers[i](x, training=True)
    # group conv
    x = model.layers[96](x, 2, axis=-1)
    tmp0 = model.layers[97](x[0])
    tmp1 = model.layers[98](x[1])
    x = model.layers[99]([tmp0, tmp1], axis=3)
    x = model.layers[100](x, x8)
    if residual:
        r8 = insert_residual(x8, x)
        x = tf.keras.layers.Add()([x, r8])

    x10_1 = x
    for i in range(101, 107):
        x = model.layers[i](x, training=True)
    if residual:
        r10_1 = insert_residual(x10_1, x)
        x = tf.keras.layers.Add()([x, r10_1])

    x11 = x
    for i in range(107, 113):
        x = model.layers[i](x, training=True)
    x = model.layers[113](x, x11)
    if residual:
        r11 = insert_residual(x11, x)
        x = tf.keras.layers.Add()([x, r11])

    x12 = x
    for i in range(114, 120):
        x = model.layers[i](x, training=True)
    x = model.layers[120](x, x12)
    if residual:
        r12 = insert_residual(x12, x)
        x = tf.keras.layers.Add()([x, r12])

    x13 = x
    for i in range(121, 127):
        x = model.layers[i](x, training=True)
    x = model.layers[127](x, x13)
    if residual:
        r13 = insert_residual(x13, x)
        x = tf.keras.layers.Add()([x, r13])

    # last layers
    for i in range(128, len(model.layers)):
        x = model.layers[i](x, training=True)

    new_model = tf.keras.Model(inputs=input, outputs=x)
    return new_model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = get_fbnet_backbone()
    model.summary()
