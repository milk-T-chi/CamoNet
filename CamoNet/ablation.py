import os
import numpy as np
import tensorflow as tf
import argparse
from Data_Loader import Loader
from Augmenter import get_augmenter
from LiteSiamese import get_encoder, LiteSiameseModel
from optimizer_utils import *
from residual_utils import *
import pandas as pd
import matplotlib.pyplot as plt
from graph2model import *

# GPU configuration

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_num', type=str, default="0")

# Dataset related Hyperparameters
parser.add_argument('--dataset_name', type=str, default="cifar10")  #
parser.add_argument('--data_root_path', type=str, default="./Split_Data/")
parser.add_argument('--unlabeled_dataset_rate', type=float, default=0.99)  #
parser.add_argument("--image_size", type=int, default=96)  #
parser.add_argument("--image_channels", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=256)  #
parser.add_argument("--num_class", type=int, default=10)  #
parser.add_argument("--total_data", type=int, default=50000)  #

# Data augmentation related Hyperparameters
parser.add_argument('--min_area', type=float, default=0.4)
parser.add_argument('--brightness', type=float, default=0.5)
parser.add_argument('--jitter', type=float, default=0.2)

# Model structure related Hyperparameters
parser.add_argument('--project_dim', type=int, default=256)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--kernel_regularizer', type=float, default=5e-4)
parser.add_argument('--hyp', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=1)  #
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
parser.add_argument('--add_residual', type=bool, default=True)  #

parser.add_argument('--warmup_rate', type=float, default=0.05)
parser.add_argument('--start_lr', type=float, default=0.0)
parser.add_argument('--target_lr', type=float, default=5e-3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--label_smoothing', type=float, default=0.3)
parser.add_argument('--dropout', type=float, default=0.3)

parser.add_argument('--contrastive_epoch', type=int, default=100)  #
parser.add_argument('--supervised_epoch', type=int, default=100)  #
parser.add_argument('--load_weights', type=bool, default=True)  #
parser.add_argument('--save_weights', type=bool, default=True)  #
parser.add_argument('--load_weights_path', type=str, default="./save_weights/cifar10/OFA/contrastive.ckpt")  #
parser.add_argument('--save_weights_path', type=str, default="./save_weights/cifar10/OFA/contrastive.ckpt")  #
# Options: contrastive, supervised_Last, supervised_Residual, supervised_All
parser.add_argument('--training_mode', type=str, default="supervised_Last")  #

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

# Load dataset
data_loader = Loader(
    dataset_name=args.dataset_name,
    data_root_path=args.data_root_path,
    unlabeled_dataset_rate=args.unlabeled_dataset_rate,
    image_size=args.image_size,
    image_channels=args.image_channels,
    batch_size=args.batch_size,
    num_class=args.num_class,
    total_data=args.total_data,
)

label_ds, unlabel_ds, test_ds = data_loader.load_data()


def visualize_augmentations(num_images, dataset):
    contrastive_augmentation = {"min_area": 0.4, "brightness": 0.5,
                                "jitter": 0.2, "image_size": 224}
    # Sample a batch from a dataset
    images = next(iter(dataset))
    images = tf.keras.layers.Rescaling(scale=1 / 255, offset=0)(images)
    # Apply augmentations
    augmented_images = zip(
        images,
        get_augmenter(**contrastive_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
    )

    row_titles = [
        "Original:",
        "Weakly augmented:",
        "Strongly augmented:",
        "Strongly augmented:",
    ]

    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(4, num_images, row * num_images + column + 1)
            plt.imshow(tf.cast(image, tf.float32))
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# visualize_augmentations(num_images=16, dataset=unlabel_ds)

# Build model
contrastive_augmentation = {"min_area": args.min_area, "brightness": args.brightness,
                            "jitter": args.jitter, "image_size": args.image_size}

# encoder = get_encoder(
#     trainable=args.backbone_trainable,
#     weights=args.weights,
#     image_size=args.image_size,
#     image_channels=args.image_channels,
#     alpha=args.alpha,
#     inverted_residual_setting=args.inverted_residual_setting)

encoder = get_ofa_backbone(shape=(args.image_size, args.image_size, 3), residual=True)
zero_initializer(encoder)
encoder.trainable = True

backbone = get_ofa_backbone(shape=(args.image_size, args.image_size, 3), residual=False)
backbone.trainable = False

for layer in encoder.layers:
    if not "siamese_residual" in layer.name:
        layer.trainable = False

# if args.add_residual:
#     encoder = insert_residual_all(model=encoder, kernel_size=args.kernel_size,
#                                   channel_reduce_ratio=args.channel_reduce_ratio,
#                                   downsample_ratio=args.downsample_ratio)
#     zero_initializer(encoder)

inputs = tf.keras.layers.Input((args.image_size, args.image_size, args.image_channels))
x = encoder(inputs)
outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
encoder = tf.keras.Model(inputs, outputs)

inputs = tf.keras.layers.Input((args.image_size, args.image_size, args.image_channels))
x = backbone(inputs)
outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
backbone = tf.keras.Model(inputs, outputs)

contrastive_model = LiteSiameseModel(encoder=encoder, backbone=backbone)
contrastive_model.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
contrastive_model.project_dim = args.project_dim
contrastive_model.latent_dim = args.latent_dim
contrastive_model.kernel_regularizer = args.kernel_regularizer
contrastive_model.hyp = args.hyp

if args.training_mode.__contains__("contrastive"):
    total_steps = (args.total_data / args.batch_size) * args.contrastive_epoch
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=args.target_lr,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=int(args.warmup_rate * total_steps),
                                            hold_base_rate_steps=0)
    contrastive_model.compile(
        contrastive_optimizer=tf.keras.optimizers.Adam(), run_eagerly=True
    )
    contrastive_model.encoder.summary()

    contrastive_history = contrastive_model.fit(
        unlabel_ds, epochs=args.contrastive_epoch, callbacks=[warm_up_lr]
    )

    if args.save_weights:
        contrastive_model.encoder.save_weights(args.save_weights_path)

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

encoder_ = get_ofa_backbone(shape=(args.image_size, args.image_size, 3), residual=True)

inputs = tf.keras.layers.Input((args.image_size, args.image_size, args.image_channels))
x = encoder_(inputs)
outputs = tf.keras.layers.GlobalAveragePooling2D()(x)

ft_model = LiteSiameseModel(encoder=tf.keras.Model(inputs, outputs), backbone=backbone)

if args.load_weights:
    ft_model.encoder.load_weights(args.load_weights_path)

if args.training_mode.__contains__("supervised"):

    ft_model.encoder.trainable = False

    inputs = tf.keras.layers.Input(shape=(args.image_size, args.image_size, args.image_channels))
    # x = tf.keras.layers.Rescaling(scale=2, offset=-1)(inputs)
    x = ft_model.encoder(inputs)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    outputs = tf.keras.layers.Dense(args.num_class)(x)

    finetuning_model = tf.keras.Model(inputs, outputs)

    finetuning_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                 tf.keras.losses.CategoricalCrossentropy(from_logits=True, name="loss")]
    )

    finetuning_model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=args.patience, restore_best_weights=True
    )

    finetuning_history = finetuning_model.fit(
        label_ds, epochs=args.supervised_epoch, validation_data=test_ds, callbacks=[early_stopping]
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(finetuning_history.history["val_acc"]) * 100
        )
    )

    if args.training_mode.__contains__("Residual") or args.training_mode.__contains__("All"):

        finetuning_model.trainable = True
        if args.training_mode.__contains__("Residual"):
            for layer in finetuning_model.layers[1].layers[1].layers:
                if not layer.name.__contains__("siamese_residual"):
                    layer.trainable = False

        finetuning_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                     tf.keras.losses.CategoricalCrossentropy(from_logits=True, name="loss")]
        )

        finetuning_model.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=args.patience, restore_best_weights=True
        )

        history = finetuning_model.fit(
            label_ds, epochs=args.supervised_epoch, validation_data=test_ds, callbacks=[early_stopping]
        )
        print(
            "Maximal validation accuracy: {:.2f}%".format(
                max(history.history["val_acc"]) * 100
            )
        )
