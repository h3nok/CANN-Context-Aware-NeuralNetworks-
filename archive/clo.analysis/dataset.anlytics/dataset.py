import tensorflow_datasets as tfds
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Dataset:
    _train = None
    _test = None
    _loader = None
    _metadata = None
    _train_total = None
    _test_total = None
    _name = None

    def __init__(self, name=None, train=None, test=None, batch_size=None):
        assert name
        self._train = train
        self._name = name
        self._test = test
        if not self._train or not self._test:
            if batch_size:
                self._loader, self._metadata = tfds.load(name, with_info=True, batch_size=batch_size)
            else:
                self._loader, self._metadata = tfds.load(name, with_info=True)
            self._train, self._test = tfds.load(name, split='train'), tfds.load(name, split='test')
            self._train_total = self._metadata.splits['train']
            self._test_total = self._metadata.splits['test']

    def get(self):
        assert isinstance(self._train, tf.data.DeepCLODataProvider)
        assert isinstance(self._test, tf.data.DeepCLODataProvider)
        return self._train, self._test

    @property
    def features(self):
        return self._metadata.features

    @property
    def num_classes(self):
        return self._metadata.features['label'].num_classes

    @property
    def labels(self):
        return self._metadata.features['label'].names

    def show_examples(self, split='train'):
        fig = None
        if split == 'train':
            fig = tfds.show_examples(self._metadata, self._train)
        else:
            fig = tfds.show_examples(self._metadata, self._test)

        return fig

    @property
    def to_numpy(self):
        return tfds.as_numpy(self._train), tfds.as_numpy(self._test)

    @property
    def info(self):
        return self._metadata

    @property
    def total_training_samples(self):
        return self._train_total

    @property
    def total_test_samples(self):
        return self._test_total

    @property
    def name(self):
        return self._name

    def ds2tfrecord(self, split, filepath):
        ds = None
        if split is 'train':
            ds = self._train
        elif split is 'test':
            ds = self._test

        with tf.io.TFRecordWriter(filepath) as writer:
            feat_dict = tf.data.make_one_shot_iterator(ds).get_next()
            serialized_dict = {name: tf.io.serialize_tensor(fea) for name, fea in feat_dict.items()}
            with tf.Session() as sess:
                try:
                    while True:
                        features = {}
                        for name, serialized_tensor in serialized_dict.items():
                            bytes_string = sess.run(serialized_tensor)
                            bytes_list = tf.train.BytesList(value=[bytes_string])
                            features[name] = tf.train.Feature(bytes_list=bytes_list)
                        # Create a Features message using tf.train.Example.
                        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                        example_string = example_proto.SerializeToString()
                        # Write to TFRecord
                        writer.write(example_string)
                except tf.errors.OutOfRangeError:
                    pass
