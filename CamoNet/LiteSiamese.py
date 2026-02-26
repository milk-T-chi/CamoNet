import tensorflow as tf
from Model import MobileNet_base
from weight_sharing.bias_utils import *
from Augmenter import *


def compute_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


def get_encoder(trainable=False, weights="imagenet", image_size=224, alpha=1, image_channels=3,
                inverted_residual_setting=[
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1]]):
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, image_channels),
        alpha=alpha,
        include_top=False,
        weights=weights,
        input_tensor=None,
        pooling=None)

    model, setting, alpha = MobileNet_base(alpha=alpha, im_height=image_size, im_width=image_size,
                                           include_top=False, inverted_residual_setting=inverted_residual_setting)

    model = reorganize_model(pretrained_model, model, setting, alpha=alpha, choice='dw')
    model.trainable = trainable
    return model


class LiteSiameseModel(tf.keras.Model):
    def __init__(self, encoder, backbone, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_augmenter = get_augmenter()
        self.encoder = encoder
        self.backbone = backbone
        self.resize_layer = tf.keras.layers.Rescaling(scale=2, offset=-1)
        self.cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)
        self.project_dim = 128
        self.latent_dim = 64
        self.kernel_regularizer = 5e-4
        self.hyp = 1.0

        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.encoder.output.shape[-1])),
                tf.keras.layers.Dense(self.project_dim, use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
                                      ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6.0),
                tf.keras.layers.Dense(self.project_dim, use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
                                      ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6.0),
            ],
            name="projection_head",
        )

        self.predictor = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.projection_head.output.shape[-1],)),
                tf.keras.layers.Dense(self.latent_dim, use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6.0),
                tf.keras.layers.Dense(self.project_dim, use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)),
                # tf.keras.layers.BatchNormalization()
            ],
            name="predictor"
        )

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        self.cosine_loss_tracker = tf.keras.metrics.Mean(name="cosine_loss")

    def train_step(self, unlabeled_images):
        augmented_images_1 = tf.keras.layers.Rescaling(scale=1 / 255, offset=0)(unlabeled_images)
        augmented_images_2 = tf.keras.layers.Rescaling(scale=1 / 255, offset=0)(unlabeled_images)
        augmented_images_1 = self.contrastive_augmenter(augmented_images_1, training=True)
        augmented_images_2 = self.contrastive_augmenter(augmented_images_2, training=True)
        augmented_images_1 = tf.keras.layers.Rescaling(scale=255, offset=0)(augmented_images_1)
        augmented_images_2 = tf.keras.layers.Rescaling(scale=255, offset=0)(augmented_images_2)
        augmented_images_1 = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_images_1)
        augmented_images_2 = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_images_2)
        # augmented_images_1 = self.resize_layer(augmented_images_1, training=True)
        # augmented_images_2 = self.resize_layer(augmented_images_2, training=True)

        unlabeled_features = tf.keras.applications.mobilenet_v2.preprocess_input(unlabeled_images)

        for layer in self.encoder.layers:
            if layer.name.__contains__("model"):
                for sublayer in layer.layers:
                    if sublayer.name.__contains__("Conv") or sublayer.name.__contains__("inverted_residual"):
                        unlabeled_features = sublayer(unlabeled_features, training=False)
            else:
                unlabeled_features = layer(unlabeled_features, training=False)

        # unlabeled_features = self.backbone(unlabeled_features, training=False)

        with tf.GradientTape() as tape:
            # 第一部分，计算对比loss
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            z1 = self.projection_head(features_1, training=True)
            z2 = self.projection_head(features_2, training=True)
            z_ = self.projection_head(unlabeled_features, training=True)

            p1 = self.predictor(z1, training=True)
            p2 = self.predictor(z2, training=True)
            p_ = self.predictor(z_, training=True)

            loss1 = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2
            loss2 = compute_loss(p1, z_) / 2 + compute_loss(p_, z1) / 2
            loss3 = compute_loss(p2, z_) / 2 + compute_loss(p_, z2) / 2

            loss = loss1 + self.hyp * (loss2 + loss3) / 2

        gradients = tape.gradient(
            loss,
            self.encoder.trainable_variables + self.projection_head.trainable_variables
            + self.predictor.trainable_variables
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_variables + self.projection_head.trainable_variables
                + self.predictor.trainable_variables
            )
        )

        self.contrastive_loss_tracker.update_state(loss1)
        self.cosine_loss_tracker.update_state((loss2 + loss3) / 2)
        return {m.name: m.result() for m in self.metrics}
