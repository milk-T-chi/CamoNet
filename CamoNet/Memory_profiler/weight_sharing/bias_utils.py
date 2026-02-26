from weight_sharing import reorganize_model
from Model import *
from residual_utils import *


def set_bias_trainable(model):
    model.trainable = True
    for layer in model.layers:
        if str(type(layer)).__contains__("InvertedResidual") or str(type(layer)).__contains__("ConvBNReLU"):
            sublayer_list = layer.get_all_layers()
            for sub_layer in sublayer_list:
                if not sub_layer.name.__contains__("bias"):
                    sub_layer.trainable = False


def set_bias_res_trainable(model):
    model.trainable = True
    for layer in model.layers:
        if layer.name.__contains__("model"):
            for layer_ in layer.layers:
                if layer_.name.__contains__("model"):
                    for sublayer in layer_.layers:
                        if str(type(sublayer)).__contains__("InvertedResidual") or str(type(sublayer)).__contains__(
                                "ConvBNReLU"):
                            sublayer_list = sublayer.get_all_layers()
                            for sub_layer_ in sublayer_list:
                                if not sub_layer_.name.__contains__("bias"):
                                    sub_layer_.trainable = False


def zero_initializer(model):
    initializer = tf.keras.initializers.RandomUniform(minval=-1e-20, maxval=1e-20)
    # initializer = tf.keras.initializers.Constant()

    for layer in model.layers:
        if layer.name.__contains__("bottle_neck"):
            for sublayer in layer.get_conv_layers():
                kernal_shape = sublayer.get_weights()[0].shape
                # bias_shape = sublayer.get_weights()[1].shape
                zero_param = [initializer(shape=kernal_shape)]
                sublayer.set_weights(zero_param)

        if layer.name.__contains__("lite_residual"):
            for sublayer in layer.get_conv_layers():
                kernal_shape = sublayer.get_weights()[0].shape
                zero_param = [initializer(shape=kernal_shape)]
                sublayer.set_weights(zero_param)


if __name__ == "__main__":
    alpha = 1.0
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
    )
    model, inverted_residual_setting, _ = MobileNet_base(im_height=96, im_width=96, inverted_residual_setting=[
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1], ], alpha=alpha, include_top=False)

    model = reorganize_model(pretrained_model, model, inverted_residual_setting, alpha=alpha, choice='dw')
    model.trainable = False
    set_bias_trainable(model)

    model = insert_lite_residual_all(model)

    zero_initializer(model)
    # model.summary()
