import os
import tensorflow as tf
import numpy as np
import datetime
from typing import Tuple, List

from ..utils.measurement_model import create_cs_measurement_model
from ..utils.initializer import xavier_init

# Logger
import logging
#   create logger and set level to info
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())  # http://docs.python-guide.org/en/latest/writing/logging/#logging-in-a-library
#   create console handler (ch) and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
#   create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#   add formatter to ch
ch.setFormatter(formatter)
#   add ch to logger
logger.addHandler(ch)
folder_sep = '-'
log_sep = '##################################################################################################'


class SDA:
    def __init__(self,
                 data_dim: int,
                 compression_percent: int,
                 layer_number: int = 4,
                 learn_mm: bool = False,
                 redundancy_factor: int = 1,
                 dtype: tf.DType = tf.float32,
                 base_dir: str = 'networks'):

        self.__data_dim = data_dim
        self.__compression_percent = compression_percent
        if not layer_number % 2 == 0 or layer_number < 4:
            raise TypeError('Argument `layer_number` must be even and >= 4')
        else:
            self.__layer_number = layer_number
        self.__compressed_dim = round((1 - self.compression_percent / 100) * self.data_dim)
        self.__undersampling_ratio = self.compressed_dim / self.data_dim  # exact M/N
        if not isinstance(redundancy_factor, int) or redundancy_factor < 1:
            raise TypeError('Argument `redundancy_factor` must be a `int` >= 1')
        else:
            self.__redundancy_factor = redundancy_factor
        self.__hidden_dec_dim = self.data_dim * self.redundancy_factor
        self.__hidden_enc_dim = self.compressed_dim * self.redundancy_factor
        if not isinstance(learn_mm, bool):
            raise TypeError('Argument `learn_mm` must be a `bool`')
        else:
            self.__learn_mm = learn_mm
        if not isinstance(dtype, tf.DType):
            raise TypeError('Argument `dtype` must be a `tf.DType`')
        self.__dtype = dtype

        # Model name
        self.__name = folder_sep.join(['SDA', '{nl}'.format(nl='CL' if self.learn_mm else 'CNL'),
                                       'CP{cp:d}'.format(cp=self.compression_percent),
                                       'LN{ln:d}'.format(ln=self.layer_number),
                                       'RF{rf:d}'.format(rf=self.redundancy_factor)])

        # Build model: inputs, variables, outputs
        tf.reset_default_graph()  # reset default graph (in case of multiple graphs)
        x_in = tf.placeholder(dtype=dtype, shape=[None, self.data_dim], name='inputs')
        self.__inputs = x_in
        outputs, variables = self._build_model()
        self.__variables = variables
        self.__outputs = outputs

        # Loss
        norm_diff = tf.reduce_sum(tf.square(self.outputs - self.inputs), axis=1)
        self.__loss = tf.reduce_mean(norm_diff, axis=0, name='loss')

        # Saver
        if not isinstance(base_dir, str):
            raise TypeError('Argument `base_dir` must be a `str`')
        else:
            self.__model_dir = os.path.join(base_dir, self.name)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.__model_path = os.path.join(self.model_dir, self.name)
        self.__saver = tf.train.Saver(var_list=variables, max_to_keep=5)

    # Properties
    @property
    def name(self) -> str:
        return self.__name

    @property
    def data_dim(self) -> int:
        return self.__data_dim

    @property
    def compression_percent(self) -> int:
        return self.__compression_percent

    @property
    def compressed_dim(self) -> int:
        return self.__compressed_dim

    @property
    def undersampling_ratio(self) -> float:
        return self.__undersampling_ratio

    @property
    def layer_number(self) -> int:
        return self.__layer_number

    @property
    def redundancy_factor(self) -> int:
        return self.__redundancy_factor

    @property
    def learn_mm(self) -> bool:
        return self.__learn_mm

    @property
    def hidden_dec_dim(self) -> int:
        return self.__hidden_dec_dim

    @property
    def hidden_enc_dim(self) -> int:
        return self.__hidden_enc_dim

    @property
    def dtype(self) -> tf.DType:
        return self.__dtype

    @property
    def inputs(self) -> tf.Tensor:
        return self.__inputs

    @property
    def variables(self) -> List[tf.Variable]:
        return self.__variables

    @property
    def outputs(self) -> tf.Tensor:
        return self.__outputs

    @property
    def model_dir(self) -> str:
        return self.__model_dir

    @property
    def model_path(self) -> str:
        return self.__model_path

    @property
    def loss(self):
        return self.__loss

    @property
    def saver(self):
        return self.__saver

    # Methods
    def evaluate(self, sess, inputs):
        outputs = sess.run(self.outputs, feed_dict={self.inputs: inputs})
        return outputs

    def train(self,
              learning_rate: float,
              train_set: np.ndarray,
              valid_set: np.ndarray = None,
              num_epochs: int = 1,
              batch_size: int = 1,
              optimizer=tf.train.AdamOptimizer,
              dump_percent: int = 0,
              save: bool = True,
              save_suffix=None,
              save_check: bool = True) -> None:
        # Input check
        #   Check if directory already contains a saved network
        if save_suffix is not None:
            self._append_name_suffix(suffix=save_suffix)
        if save_check:  # Can be used to bypass folder check
            if save and os.path.exists(self.model_dir):
                if os.listdir(self.model_dir):  # check not empty dir
                    usr_in = input('Directory `{path}` already contains a saved model, '
                                   'do you want to continue (y/[n])? '
                                   .format(path=self.model_dir))
                    usr_in = usr_in.lower() if usr_in else 'n'  # default: 'n'
                    if usr_in.lower() == 'n':
                        raise InterruptedError
        #   Logger / dumper
        if not isinstance(dump_percent, int) or dump_percent > 100 or dump_percent < 0:
            raise TypeError('Argument `dump_percent` must be a `int` >=0 and <=100')

        #   Batch size
        train_set_size = train_set.shape[0]
        if batch_size > train_set_size:
            batch_size = train_set_size
            logger.warning('Batch size must but smaller or equal to train set size, truncated to train size')

        #   Valid set
        valid_mean_loss = 'NO VALID SET'
        if valid_set is None:
            valid_set_size = 'NO VALID SET'
        else:
            valid_set_size = valid_set.shape[0]

        # Optimize operation
        opt = optimizer(learning_rate=learning_rate).minimize(self.loss)

        # Initializer
        init = tf.global_variables_initializer()

        # Start Session
        with tf.Session() as sess:
            # Initialization
            sess.run(init)
            logger.info(log_sep)
            logger.info('NETWORK INITIALIZED')
            logger.info('  Name: {}'.format(self.name))
            logger.info('  Train set size: {:d}'.format(train_set_size))
            logger.info('  Valid set size: {}'.format(valid_set_size))
            logger.info(log_sep)

            # Init epochs
            seed = 123456789
            prng = np.random.RandomState(seed=seed)
            epoch_steps = int(train_set_size / batch_size)
            for epoch in range(num_epochs):
                t_epoch_start = datetime.datetime.now()
                # Random shuffle training set
                prng.shuffle(train_set)
                if dump_percent is None or dump_percent == 0:
                    dump_ratio = epoch_steps
                else:
                    dump_ratio = round(dump_percent / 100 * epoch_steps)
                for step in range(epoch_steps):
                    # Pick an offset within the training data, which has been randomized.
                    offset = (step * batch_size) % (train_set_size - batch_size)

                    # Get a minibatch.
                    batch_data = train_set[offset:(offset + batch_size), :]

                    # Run optimizer on the minibatch and get the mean loss
                    _, train_loss_val = sess.run([opt, self.loss], feed_dict={self.inputs: batch_data})

                    # Logger
                    #   Always dump first and last step of an epoch
                    log_step = 'Epoch number: {:5d}/{:d}, step: {:5d}/{:d}, train loss: {:>12.6f}'.format(
                        epoch + 1, num_epochs, step + 1, epoch_steps, train_loss_val)
                    if step == 0 or step == epoch_steps - 1:
                        logger.info(log_step)
                    elif step % dump_ratio == 0:
                        logger.info(log_step)

                # Validation set
                if valid_set is not None:
                    valid_set_size = valid_set.shape[0]
                    valid_steps = int(valid_set_size / batch_size)
                    if valid_steps == 0:
                        valid_mean_loss = sess.run(self.loss, feed_dict={self.inputs: valid_set})
                    else:
                        valid_loss_list = []
                        for valid_step in range(valid_steps):
                            # Pick an offset within the training data, which has been randomized.
                            offset = (valid_step * batch_size) % (valid_set_size - batch_size)
                            # Generate a minibatch.
                            batch_in = valid_set[offset:(offset + batch_size), :]
                            valid_loss = sess.run(self.loss, feed_dict={self.inputs: batch_in})
                            valid_loss_list.append(valid_loss)
                        valid_mean_loss = np.mean(np.array(valid_loss_list))

                # Save after each epoch
                logger.info('Epoch number: {:5d}/{:d}'.format(epoch + 1, num_epochs))
                logger.info('  Valid mean loss: {}'.format(valid_mean_loss))
                self.save(sess=sess, global_step=epoch + 1)
                logger.info('  Model saved under {:s}-{:d}'.format(self.model_path, epoch + 1))
                epoch_time = (datetime.datetime.now() - t_epoch_start)
                logger.info('  Total epoch time: {} seconds'.format(epoch_time.seconds))
                logger.info(log_sep)

    def save(self, sess, global_step:int = None) -> None:
        _ = self.saver.save(sess=sess, save_path=self.model_path, global_step=global_step)

    def restore(self, sess, suffix=None) -> None:
        # Update name with suffix if provided
        self._append_name_suffix(suffix=suffix)
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

    def _append_name_suffix(self, suffix=None):
        if suffix:  # not empty (i.e. not empty string or None)
            # Update name, model_dir (create), model_path
            self.__name = folder_sep.join([self.name, '{suffix}'.format(suffix=suffix)])
            self.__model_dir = folder_sep.join([self.model_dir, '{suffix}'.format(suffix=suffix)])
            self.__model_path = os.path.join(self.model_dir, self.name)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def _build_model(self) -> Tuple[tf.Tensor, List[tf.Variable]]:
        #   Layer 1, i.e. compression layer
        with tf.name_scope('layer1_cmp'):
            # Depends if the compression is learned (SDA-CL) or not (SDA-CNL)
            if not self.learn_mm:
                mm = create_cs_measurement_model(type='gaussian-orth', input_dim=self.data_dim,
                                                 compression_percent=self.compression_percent)
                W_in = tf.constant(value=mm.T, dtype=self.dtype)
                y_in = tf.matmul(self.inputs, W_in)
            else:
                W_in = tf.Variable(xavier_init(fan_in=self.data_dim, fan_out=self.compressed_dim),
                                   dtype=self.dtype, name='weights')
                b_in = tf.Variable(tf.zeros([self.compressed_dim], dtype=self.dtype), name='biases')
                y_in = tf.nn.tanh(tf.matmul(self.inputs, W_in) + b_in, name='outputs')

        #   Layer 2, i.e. decompression + eventual dimensionality expansion (if `redundancy_factor > 1`)
        with tf.name_scope('layer2_dec'):
            W_dec = tf.Variable(xavier_init(fan_in=self.compressed_dim, fan_out=self.hidden_dec_dim),
                                dtype=self.dtype, name='weights')
            b_dec = tf.Variable(tf.zeros([self.hidden_dec_dim], dtype=self.dtype), name='biases')
            y_dec = tf.nn.tanh(tf.matmul(y_in, W_dec) + b_dec, name='outputs')

        #   Layers 3 to `layer_number - 2`, i.e. encode -> decode -> encode -> decode -> etc.
        for de_pair_nb in range(2, self.layer_number - 2, 2):
            with tf.name_scope('layer{ln:d}_enc'.format(ln=de_pair_nb + 1)):
                W_enc = tf.Variable(xavier_init(fan_in=self.hidden_dec_dim, fan_out=self.hidden_enc_dim),
                                    dtype=self.dtype, name='weights')
                b_enc = tf.Variable(tf.zeros([self.hidden_enc_dim], dtype=self.dtype), name='biases')
                y_enc = tf.nn.tanh(tf.matmul(y_dec, W_enc) + b_enc, name='outputs')
            with tf.name_scope('layer{ln:d}_dec'.format(ln=de_pair_nb + 2)):
                W_dec = tf.Variable(xavier_init(fan_in=self.hidden_enc_dim, fan_out=self.hidden_dec_dim),
                                    dtype=self.dtype, name='weights')
                b_dec = tf.Variable(tf.zeros([self.hidden_dec_dim], dtype=self.dtype), name='biases')
                y_dec = tf.nn.tanh(tf.matmul(y_enc, W_dec) + b_dec, name='outputs')

        #   Layer N-1
        with tf.name_scope('layer{ln:d}_enc'.format(ln=self.layer_number - 1)):
            W_enc = tf.Variable(xavier_init(fan_in=self.hidden_dec_dim, fan_out=self.hidden_enc_dim),
                                dtype=self.dtype, name='weights')
            b_enc = tf.Variable(tf.zeros([self.hidden_enc_dim], dtype=self.dtype), name='biases')
            y_enc = tf.nn.tanh(tf.matmul(y_dec, W_enc) + b_enc, name='outputs')

        #   Output layer
        with tf.name_scope('layer{ln:d}_out'.format(ln=self.layer_number)):
            W_out = tf.Variable(xavier_init(fan_in=self.hidden_enc_dim, fan_out=self.data_dim),
                                dtype=self.dtype, name='weights')
            b_out = tf.Variable(tf.zeros([self.data_dim], dtype=self.dtype), name='biases')
            y_out = tf.nn.tanh(tf.matmul(y_enc, W_out) + b_out, name='outputs')

        variables = tf.global_variables()

        return y_out, variables
