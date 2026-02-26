import os
import numpy as np
import tensorflow as tf
import argparse
from Data_Loader import Loader
from LiteSiamese import get_encoder, LiteSiameseModel, ProjectionPredictor
from optimizer_utils import WarmUpCosineDecay
from residual_utils import *
from tqdm import tqdm
from itertools import combinations
from itertools import product
from random import sample
from keras_flops import get_flops
from memory_cost_v3 import profile_memory_cost
from memory_cost_v3 import count_param_size
import pandas as pd
import time
from HiddenPrints import suppress_stdout_stderr

# GPU configuration

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_num', type=str, default="-1")

# Dataset related Hyperparameters
parser.add_argument("--image_channels", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=128)  #
parser.add_argument("--num_class", type=int, default=10)  #
parser.add_argument("--total_data", type=int, default=50000)  #

# Data augmentation related Hyperparameters
parser.add_argument('--min_area', type=float, default=0.4)
parser.add_argument('--brightness', type=float, default=0.5)
parser.add_argument('--jitter', type=float, default=0.2)

# Model structure related Hyperparameters
parser.add_argument('--project_dim', type=int, default=128)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--kernel_regularizer', type=float, default=5e-4)
parser.add_argument('--hyp', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=1.0)  #
parser.add_argument('--inverted_residual_setting', type=list,
                    default=[[1, 16, 1, 1],
                             [6, 24, 2, 2],
                             [6, 32, 3, 2],
                             [6, 64, 4, 2],
                             [6, 96, 3, 1],
                             [6, 160, 3, 2],
                             [6, 320, 1, 1], ])  #
parser.add_argument('--weights', type=str, default="imagenet")
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--downsample_ratio', type=int, default=2)
parser.add_argument('--channel_reduce_ratio', type=int, default=1)

# Training related Hyperparameters
parser.add_argument('--backbone_trainable', type=bool, default=False)
parser.add_argument('--add_residual', type=bool, default=True)

parser.add_argument('--warmup_rate', type=float, default=0.05)
parser.add_argument('--start_lr', type=float, default=0.0)
parser.add_argument('--target_lr', type=float, default=5e-3)  #

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--label_smoothing', type=float, default=0.3)
parser.add_argument('--dropout', type=float, default=0.3)

parser.add_argument('--contrastive_epoch', type=int, default=200)  #
parser.add_argument('--supervised_epoch', type=int, default=200)
parser.add_argument('--load_weights', type=bool, default=True)  #
parser.add_argument('--save_weights', type=bool, default=True)  #
parser.add_argument('--load_weights_path', type=str, default="./save_weights/cifar10/1/contrastive.ckpt")  #
parser.add_argument('--save_weights_path', type=str, default="./save_weights/cifar10/1/contrastive.ckpt")  #
# Options: contrastive, supervised_Last, supervised_Residual, supervised_All
parser.add_argument('--training_mode', type=str, default="contrastive+supervised_Last")  #

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num


t_config_1 = [1]
t_config_6 = [3, 4, 6]
n_config_1 = [1]
n_config_2 = [2]
n_config_3 = [2, 3]
n_config_4 = [2, 3, 4]

total_config_t_n = []
for t in product(t_config_1, t_config_6, t_config_6, t_config_6, t_config_6, t_config_6, t_config_6, n_config_1,
                 n_config_2, n_config_3, n_config_4, n_config_3, n_config_3, n_config_1):
    total_config_t_n.append(list(t))
# total_config_t_n_now = sample(total_config_t_n, 2000)
c_list = [16, 24, 32, 64, 96, 160, 320]
s_list = [1, 2, 2, 2, 1, 2, 1]

# Build model
# contrastive_augmentation = {"min_area": args.min_area, "brightness": args.brightness,
#                             "jitter": args.jitter, "image_size": args.image_size}

# contrastive_model = LiteSiameseModel()
# contrastive_model.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
# contrastive_model.project_dim = args.project_dim
# contrastive_model.latent_dim = args.latent_dim
# contrastive_model.kernel_regularizer = args.kernel_regularizer
# contrastive_model.hyp = args.hyp
# total_image_size_list = [224, 192, 160, 128, 96]
total_image_size_list = [224]
small_image_size_list = [224]
alpha_list = [1.0]
# lite_residual_list = []
image_channels = 3
batch_size = 16
save_path = './FLOPs_Memory/'
for i in range(len(alpha_list)):
    alpha_now = alpha_list[i]
    if i == 0 or i == 1:
        image_size_list = small_image_size_list
    else:
        image_size_list = total_image_size_list
    for image_size in image_size_list:
        lite_residual_config = "insert_residual"
        model_name = "mobilenet_v2_" + str(alpha_now) + "_" + str(image_size) + "_" + lite_residual_config
        print(model_name)
        total_config_t_n_now = sample(total_config_t_n, 1)
        memory_cost_list = []
        param_size_list = []
        activation_size_list = []
        batchsize_list = []
        flops_list = []
        for k in tqdm(range(len(total_config_t_n_now))):
            # start_time = time.time()
            inverted_residual_setting_now = np.ones((7, 4), dtype=np.int32)
            inverted_residual_setting_now[:, 0] = total_config_t_n_now[k][0:7]
            inverted_residual_setting_now[:, 1] = c_list
            inverted_residual_setting_now[:, 2] = total_config_t_n_now[k][7:]
            inverted_residual_setting_now[:, 3] = s_list
            inverted_residual_setting = inverted_residual_setting_now.tolist()
            # print(inverted_residual_setting_now.tolist())
            tf.keras.backend.clear_session()
            encoder = get_encoder(
                trainable=False,
                weights="imagenet",
                image_size=image_size,
                image_channels=image_channels,
                alpha=alpha_now,
                inverted_residual_setting=inverted_residual_setting)
            encoder_size, encoder_activation_size = profile_memory_cost(encoder,batch_size=1)
            # print(encoder_size, encoder_activation_size)
            with suppress_stdout_stderr():
                flops = get_flops(encoder,batch_size=1)
            # print(flops)
            backbone_encoder = insert_residual_all(model=encoder, kernel_size=args.kernel_size,
                                    channel_reduce_ratio=args.channel_reduce_ratio,
                                    downsample_ratio=args.downsample_ratio)
            backbone_encoder_size, backbone_encoder_activation_size = profile_memory_cost(backbone_encoder,batch_size=1)
            # print(backbone_encoder_size, backbone_encoder_activation_size)
            all_encoder_param_size = encoder_activation_size['param_size'] + backbone_encoder_activation_size['param_size']
            all_encoder_act_size = encoder_activation_size['act_size'] + backbone_encoder_activation_size['act_size'] * 2
            all_encoder_memory_size = all_encoder_param_size + all_encoder_act_size
            # print("memory",all_encoder_param_size + all_encoder_act_size, "param", all_encoder_param_size, "act_size",all_encoder_act_size)
            # print("flops",flops)
            encoder_output_shape = backbone_encoder.layers[-1].output_shape
            # print("backbone_encoder_output_shape", encoder_output_shape)
            projection_predictor = ProjectionPredictor(encoder_output_shape)
            inputs = tf.keras.layers.Input(shape=(encoder_output_shape[-1]))
            outputs = projection_predictor(inputs)
            PPModel = tf.keras.Model(inputs, outputs, name="projection_predictor")
            PPModel.trainable = True
            # PPModel.summary()
            _, projection_predictor_activation_size = profile_memory_cost(PPModel,batch_size=1)
            projection_predictor_param_size = projection_predictor_activation_size['param_size']
            projection_predictor_act_size = projection_predictor_activation_size['act_size'] * 3
            projection_predictor_memory_size = projection_predictor_param_size + projection_predictor_act_size
            memory_cost_list.append(all_encoder_memory_size + projection_predictor_memory_size)
            param_size_list.append(all_encoder_param_size + projection_predictor_param_size)
            activation_size_list.append(all_encoder_act_size + projection_predictor_act_size)
            flops_list.append(flops)
            # print("计算一个模型的时间",time.time() - start_time)
        model_config_now_df = pd.DataFrame({'t_n':total_config_t_n_now,'memory_cost_MB':memory_cost_list,'param_size_MB':param_size_list,'activation_size_MB':activation_size_list,'FLOPs':flops_list})
        # model_config_now_df = pd.DataFrame({'t_n':total_config_t_n_now,'memory_cost_MB':[0] * len(total_config_t_n_now)})
        model_config_now_df.to_csv(save_path + model_name + '.csv',index=False)
    #     break
    # break
        # print(profile_memory_cost(backbone_encoder, require_backward=True, activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=32))
        # print(encoder.summary())
        # pred_batch = 1
        # flops = get_flops(encoder, pred_batch)
        # print(flops)
        # print(stats_graph(encoder))
        # break
        # total_steps = (args.total_data / args.batch_size) * args.contrastive_epoch
        # schedule = WarmUpCosineDecay(start_lr=args.start_lr,
        #                             target_lr=args.target_lr,
        #                             warmup_steps=int(args.warmup_rate * total_steps),
        #                             total_steps=total_steps,
        #                             hold=int(args.warmup_rate * total_steps))
        # contrastive_model.compile(
        #     contrastive_optimizer=tf.keras.optimizers.Adam(learning_rate=schedule),
        # )
        # contrastive_model.encoder.summary()
        # contrastive_model.projection_head.summary()
        # contrastive_model.predictor.summary()

        # print(count_model_size(baseline_model,trainable_param_bits=8, frozen_param_bits=8, print_log=True))
        # a, b =count_activation_size(baseline_model, require_backward=True, activation_bits=32)
        # memory_cost_MB,param_activation_MB = profile_memory_cost(baseline_model, require_backward=True,
        #                         activation_bits=32, trainable_param_bits=32, frozen_param_bits=32, batch_size=batch_size)
        # 

    #     memory_cost_list.append(memory_cost_MB)
    #     param_size_list.append(param_activation_MB['param_size'])
    #     activation_size_list.append(param_activation_MB['act_size'])
    #     batchsize_list.append(batch_size)
    #     flops_list.append(flops)
    #     # break


    # contrastive_model.fit(
    #     unlabel_ds, epochs=args.contrastive_epoch
    # )

    # if args.save_weights:
    #     contrastive_model.encoder.save_weights(args.save_weights_path)

# ft_model = LiteSiameseModel()
# encoder_ = get_encoder(
#     trainable=args.backbone_trainable,
#     weights=args.weights,
#     image_size=args.image_size,
#     image_channels=args.image_channels,
#     alpha=args.alpha,
#     inverted_residual_setting=args.inverted_residual_setting)

# if args.add_residual:
#     encoder_ = insert_residual_all(model=encoder_, kernel_size=args.kernel_size,
#                                    channel_reduce_ratio=args.channel_reduce_ratio,
#                                    downsample_ratio=args.downsample_ratio)
#     zero_initializer(encoder)

# inputs = tf.keras.layers.Input((args.image_size, args.image_size, args.image_channels))
# x = encoder_(inputs)
# outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
# ft_model.encoder = tf.keras.Model(inputs, outputs)

# if args.load_weights:
#     ft_model.encoder.load_weights(args.load_weights_path)

# if args.training_mode.__contains__("supervised"):
#     if args.training_mode.__contains__("Last"):
#         ft_model.encoder.trainable = False
#     if args.training_mode.__contains__("All"):
#         ft_model.encoder.trainable = True

#     inputs = tf.keras.layers.Input(shape=(args.image_size, args.image_size, args.image_channels))
#     x = tf.keras.layers.Rescaling(scale=2, offset=-1)(inputs)
#     x = ft_model.encoder(x)
#     x = tf.keras.layers.Dropout(args.dropout)(x)
#     outputs = tf.keras.layers.Dense(args.num_class)(x)

#     finetuning_model = tf.keras.Model(inputs, outputs)

#     finetuning_model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
#         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing),
#         metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
#                  tf.keras.losses.CategoricalCrossentropy(from_logits=True, name="loss")]
#     )

#     finetuning_model.summary()

#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor="loss", patience=args.patience, restore_best_weights=True
#     )

#     finetuning_history = finetuning_model.fit(
#         label_ds, epochs=args.supervised_epoch, validation_data=test_ds, callbacks=[early_stopping]
#     )
#     print(
#         "Maximal validation accuracy: {:.2f}%".format(
#             max(finetuning_history.history["val_acc"]) * 100
#         )
#     )
