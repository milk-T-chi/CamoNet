import os
import tensorflow as tf
from Augmenter import get_augmenter
from optimizer_utils import *
from graph2model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
image_size = 224
image_channels = 3
batch_size = 32

dataset_name = "tf_flowers"  #
unlabeled_dataset_rate = 0.99  #
num_class = 5  #
total_data = 2752  #
contrastive_epoch = 0
target_lr = 5e-3
warmup_rate = 0.05

data_root_path = "./Split_Data/"

# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.4, "brightness": 0.5, "jitter": 0.2}


class Loader:
    def __init__(self, dataset_name, data_root_path, unlabeled_dataset_rate, image_size,
                 image_channels, batch_size, num_class, total_data,
                 **kwargs):
        super(Loader, self).__init__(**kwargs)

        self.dataset_name = dataset_name
        self.data_root_path = data_root_path
        self.unlabeled_dataset_rate = unlabeled_dataset_rate
        self.image_size = image_size
        self.image_channels = image_channels
        self.batch_size = batch_size
        self.num_class = num_class
        self.total_data = total_data

    def process_unlabeled_img(self, dataset):
        image = tf.image.resize(dataset["data"], [self.image_size, self.image_size])
        # image = tf.image.convert_image_dtype(image, tf.int8)
        return image

    def process_label_data(self, dataset):
        label = tf.one_hot(dataset["label"], depth=self.num_class)
        # image = tf.image.convert_image_dtype(dataset["data"], tf.float32)
        image = tf.image.resize(dataset["data"], [self.image_size, self.image_size])
        image = tf.keras.applications.mobilenet.preprocess_input(image)
        # image = (image - 0.5) * 2.0
        return image, label

    def _parse_function(self, example_proto):
        features = {"data": tf.io.FixedLenFeature((), tf.string),
                    "height": tf.io.FixedLenFeature((), tf.int64),
                    "width": tf.io.FixedLenFeature((), tf.int64),
                    "label": tf.io.FixedLenFeature((), tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_features['data'] = tf.reshape(tf.io.decode_raw(parsed_features['data'], tf.uint8),
                                             [parsed_features['height'], parsed_features['width'], 3])
        return parsed_features

    def load_data(self):
        labeled_data_path = self.data_root_path + self.dataset_name + "/train/supervised_train/"
        unlabeled_data_path = self.data_root_path + self.dataset_name + "/train/unsupervised_train/"
        test_data_path = self.data_root_path + self.dataset_name + "/test/"

        # unlabeled dataset
        unlabeled_srcfile = unlabeled_data_path + '100.tfrecords'
        unlabeled_train_dataset = tf.data.TFRecordDataset(unlabeled_srcfile)  # load tfrecord file
        unlabeled_train_dataset = unlabeled_train_dataset.map(self._parse_function)  # parse data into tensors

        unlabeled_train_dataset = (
            unlabeled_train_dataset
            .map(self.process_unlabeled_img, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(10000)
            .batch(self.batch_size)
        )

        # labeled dataset
        labeled_srcfile = labeled_data_path + str(int(100 - self.unlabeled_dataset_rate * 100)) + '.tfrecords'
        labeled_train_dataset = tf.data.TFRecordDataset(labeled_srcfile)  # load tfrecord file
        labeled_train_dataset = labeled_train_dataset.map(self._parse_function)  # parse data into tensors

        labeled_train_dataset = (
            labeled_train_dataset
            .map(self.process_label_data, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(10000)
            .batch(self.batch_size)
        )
        # test dataset
        test_srcfile = test_data_path + "test.tfrecords"
        test_dataset = tf.data.TFRecordDataset(test_srcfile)  # load tfrecord file
        test_dataset = test_dataset.map(self._parse_function)  # parse data into tensors
        test_dataset = (
            test_dataset
            .map(self.process_label_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return labeled_train_dataset, unlabeled_train_dataset, test_dataset


data_loader = Loader(
    dataset_name=dataset_name,
    data_root_path=data_root_path,
    unlabeled_dataset_rate=unlabeled_dataset_rate,
    image_size=image_size,
    image_channels=image_channels,
    batch_size=batch_size,
    num_class=num_class,
    total_data=total_data,
)

label_ds, unlabel_ds, test_ds = data_loader.load_data()

# model = tf.keras.applications.VGG16(
#     input_shape=(image_size, image_size, image_channels),
#     # alpha=1,
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     pooling=None)

# model = tf.keras.models.load_model("fbnet_a_backbone.h5")

model = get_fbnet_backbone(shape=(image_size, image_size, 3), residual=False)
inputs = tf.keras.layers.Input((image_size, image_size, image_channels))
x = model(inputs)
outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
encoder = tf.keras.Model(inputs, outputs)
encoder.trainable = True

encoder.summary()

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


class LiteSiameseModel(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_augmenter = get_augmenter()
        self.encoder = model
        self.resize_layer = tf.keras.layers.Rescaling(scale=2, offset=-1)
        self.cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)
        self.project_dim = 128
        self.latent_dim = 64
        self.kernel_regularizer = 5e-4

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
        # self.cosine_loss_tracker = tf.keras.metrics.Mean(name="cosine_loss")

    def train_step(self, unlabeled_images):
        augmented_images_1 = tf.keras.layers.Rescaling(scale=1 / 255, offset=0)(unlabeled_images)
        augmented_images_2 = tf.keras.layers.Rescaling(scale=1 / 255, offset=0)(unlabeled_images)
        augmented_images_1 = self.contrastive_augmenter(augmented_images_1, training=True)
        augmented_images_2 = self.contrastive_augmenter(augmented_images_2, training=True)
        augmented_images_1 = tf.keras.layers.Rescaling(scale=255, offset=0)(augmented_images_1)
        augmented_images_2 = tf.keras.layers.Rescaling(scale=255, offset=0)(augmented_images_2)
        augmented_images_1 = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_images_1)
        augmented_images_2 = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_images_2)

        # unlabeled_features = self.resize_layer(unlabeled_images, training=True)

        # for layer in self.encoder.layers:
        #     if layer.name.__contains__("model"):
        #         for sublayer in layer.layers:
        #             if sublayer.name.__contains__("Conv") or sublayer.name.__contains__("inverted_residual"):
        #                 unlabeled_features = sublayer(unlabeled_features, training=False)
        #     else:
        #         unlabeled_features = layer(unlabeled_features, training=False)

        with tf.GradientTape() as tape:
            # 第一部分，计算对比loss
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            z1 = self.projection_head(features_1, training=True)
            z2 = self.projection_head(features_2, training=True)
            # z_ = self.projection_head(unlabeled_features, training=True)

            p1 = self.predictor(z1, training=True)
            p2 = self.predictor(z2, training=True)
            # p_ = self.predictor(z_, training=True)

            loss1 = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2
            # loss2 = compute_loss(p1, z_) / 2 + compute_loss(p_, z1) / 2
            # loss3 = compute_loss(p2, z_) / 2 + compute_loss(p_, z2) / 2

            loss = loss1
            # + self.hyp * (loss2 + loss3) / 2

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
        # self.cosine_loss_tracker.update_state((loss2 + loss3) / 2)
        return {m.name: m.result() for m in self.metrics}


contrastive_model = LiteSiameseModel(encoder)
contrastive_model.contrastive_augmenter = get_augmenter(**contrastive_augmentation)

total_steps = (total_data / batch_size) * contrastive_epoch
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=target_lr,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=int(warmup_rate * total_steps),
                                        hold_base_rate_steps=0)
contrastive_model.compile(
    contrastive_optimizer=tf.keras.optimizers.Adam()
)
contrastive_model.encoder.summary()

contrastive_history = contrastive_model.fit(
    unlabel_ds, epochs=contrastive_epoch, callbacks=[warm_up_lr]
)

##########################################################
contrastive_model.encoder.trainable = False
inputs = tf.keras.layers.Input((image_size, image_size, 3))
x = contrastive_model.encoder(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_class)(x)

finetuning_model = tf.keras.Model(inputs, outputs)

finetuning_model.summary()

finetuning_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.3),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
             tf.keras.losses.CategoricalCrossentropy(from_logits=True, name="loss")]
)

finetuning_model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=10, restore_best_weights=True
)

finetuning_history = finetuning_model.fit(
    label_ds, epochs=100, validation_data=test_ds, callbacks=[early_stopping]
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(finetuning_history.history["val_acc"]) * 100
    )
)
