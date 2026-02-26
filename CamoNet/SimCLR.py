import os
import tensorflow as tf
from Augmenter import get_augmenter
from optimizer_utils import *
import helpers
from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）

image_size = 224
image_channels = 3
batch_size = 16
negative_mask = helpers.get_negative_mask(batch_size)

dataset_name = "tf_flowers"  #
unlabeled_dataset_rate = 0.95  #
num_class = 5  #
total_data = 2752  #
contrastive_epoch = 100
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
        image = tf.image.convert_image_dtype(dataset["data"], tf.float32)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image

    def process_label_data(self, dataset):
        label = tf.one_hot(dataset["label"], depth=self.num_class)
        image = tf.image.convert_image_dtype(dataset["data"], tf.float32)
        image = tf.image.resize(image, [self.image_size, self.image_size])
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
            .batch(self.batch_size, drop_remainder=True)
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

# encoder = tf.keras.applications.MobileNet(
#     input_shape=(image_size, image_size, image_channels),
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     pooling=None)
encoder = tf.keras.models.load_model("ofa_tx2.h5")
inputs = tf.keras.layers.Input((image_size, image_size, image_channels))
outputs = encoder(inputs)
# outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
encoder = tf.keras.Model(inputs, outputs)
encoder.trainable = True


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
        self.temperature = 0.1
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

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

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        # self.cosine_loss_tracker = tf.keras.metrics.Mean(name="cosine_loss")

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        l_pos = sim_func_dim1(projections_1, projections_2)
        l_pos = tf.reshape(l_pos, (l_pos.shape[0], 1))

        l_pos /= self.temperature

        negatives = tf.concat([projections_1, projections_2], axis=0)
        loss = 0

        for positives in [projections_1, projections_2]:
            l_neg = sim_func_dim2(positives, negatives)
            labels = tf.zeros(batch_size, dtype=tf.int32)
            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= self.temperature
            logits = tf.concat([l_pos, l_neg], axis=1)

            loss += self.criterion(labels, logits)
        loss = loss / (2 * batch_size)

        return loss

    def train_step(self, unlabeled_images):
        augmented_images_1 = self.contrastive_augmenter(unlabeled_images, training=True)
        augmented_images_2 = self.contrastive_augmenter(unlabeled_images, training=True)
        augmented_images_1 = self.resize_layer(augmented_images_1, training=True)
        augmented_images_2 = self.resize_layer(augmented_images_2, training=True)

        with tf.GradientTape() as tape:
            # 第一部分，计算对比loss
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            z1 = self.projection_head(features_1, training=True)
            z2 = self.projection_head(features_2, training=True)

            loss1 = self.contrastive_loss(z1, z2)

            loss = loss1

        gradients = tape.gradient(
            loss,
            self.encoder.trainable_variables + self.projection_head.trainable_variables
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_variables + self.projection_head.trainable_variables
            )
        )

        self.contrastive_loss_tracker.update_state(loss1)
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
    contrastive_optimizer=tf.keras.optimizers.Adam(), run_eagerly=True
)
contrastive_model.encoder.summary()

contrastive_history = contrastive_model.fit(
    unlabel_ds, epochs=contrastive_epoch, callbacks=[warm_up_lr]
)
