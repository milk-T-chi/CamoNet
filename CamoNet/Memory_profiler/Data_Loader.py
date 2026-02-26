import tensorflow as tf
import tensorflow_datasets as tfds


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
