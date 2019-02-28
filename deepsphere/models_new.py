"""
This module implements the graph convolutional neural network.

Most of the code is based on https://github.com/mdeff/cnn_graph/.
"""
from __future__ import division

import os
import time
import collections
import shutil
from builtins import range

import numpy as np
from scipy import sparse
import sklearn
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from . import utils


# Python 2 compatibility.
if hasattr(time, 'process_time'):
    process_time = time.process_time
else:
    import warnings
    warnings.warn('The CPU time is not working with Python 2.')
    def process_time():
        return np.nan

# def show_all_variables():
#     import tensorflow as tf
#     import tensorflow.contrib.slim as slim
#     model_vars = tf.trainable_variables()
#     slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# This class is necessary for the dataset
class LoadableGenerator(object):
    def __init__(self):
        self.curr = None
        self.it = None

    def iter(self):
        return self.__iter__()
    def __iter__(self):
        self.update()
        while self.curr:
            yield self.curr
    def load(self, it):
        self.it = it
    def update(self):
        if self.it:
            try:
                self.curr = next(self.it)
            except StopIteration:
                self.curr = None

class base_model(object):
    """Common methods for all models."""

    def __init__(self):
        self.regularizers = []
        self.regularizers_size = []

    # High-level interface which runs the constructed computational graph.

    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_training: False}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_cpu, t_wall = process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\nCPU time: {:.0f}s, wall time: {:.0f}s'.format(process_time()-t_cpu, time.time()-t_wall)
        return string, accuracy, f1, loss

    def fit(self, train_dataset, val_dataset, use_tf_dataset=False):

        # Load the dataset
        if use_tf_dataset:
            self.loadable_generator.load(train_dataset.iter(self.batch_size))

        t_cpu, t_wall = process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        if self.debug:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')

        # Initialization
        sess.run(self.op_init)

        # Training.
        accuracies_validation = []
        losses_validation = []
        losses_training = []
        num_steps = int(self.num_epochs * train_dataset.N / self.batch_size)
        if not use_tf_dataset:
            train_iter = train_dataset.iter(self.batch_size)
        else:
            sess.run(self.tf_data_iterator.initializer)

        val_data, val_labels = val_dataset.get_all_data()
        for step in range(1, num_steps+1):

            if not use_tf_dataset:
                batch_data, batch_labels = next(train_iter)
                if type(batch_data) is not np.ndarray:
                    batch_data = batch_data.toarray()  # convert sparse matrices
                feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: True}
            else:
                feed_dict = {self.ph_training: True}

            learning_rate, loss = sess.run([self.op_train, self.op_loss], feed_dict)

            evaluate = (step % self.eval_frequency == 0) or (step == num_steps)
            if evaluate and self.profile:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            learning_rate, loss = sess.run([self.op_train, self.op_loss], feed_dict, run_options, run_metadata)

            # Periodical evaluation of the model.
            if evaluate:
                epoch = step * self.batch_size / train_dataset.N
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, training loss = {:.2e}'.format(learning_rate, loss))
                losses_training.append(loss)
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies_validation.append(accuracy)
                losses_validation.append(loss)
                print('  validation {}'.format(string))
                print('  CPU time: {:.0f}s, wall time: {:.0f}s'.format(process_time()-t_cpu, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)
                if self.profile:
                    writer.add_run_metadata(run_metadata, 'step{}'.format(step))

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: best = {:.2f}, mean = {:.2f}'.format(max(accuracies_validation), np.mean(accuracies_validation[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies_validation, losses_validation, losses_training, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.

    def build_graph(self, M_0):
        """Build the computational graph of the model."""

        self.loadable_generator = LoadableGenerator()

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Make the dataset
            self.tf_train_dataset = tf.data.Dataset().from_generator(self.loadable_generator.iter, output_types=(tf.float32, tf.int32))
            self.tf_data_iterator = self.tf_train_dataset.prefetch(2).make_initializable_iterator()
            ph_data, ph_labels = self.tf_data_iterator.get_next()


            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder_with_default(ph_data, (self.batch_size, M_0), 'data')
                self.ph_labels = tf.placeholder_with_default(ph_labels, (self.batch_size), 'labels')
                self.ph_training = tf.placeholder(tf.bool, (), 'training')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_training)
            self.op_loss = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
            utils.show_all_variables()
        self.graph.finalize()

    def inference(self, data, training):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout and
            batch normalization.
            True: the model is run for training.
            False: the model is run for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, training)
        return logits

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                n_weights = np.sum(self.regularizers_size)
                regularization *= tf.add_n(self.regularizers) / n_weights
            loss = cross_entropy + regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            return loss

    def training(self, loss):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = self.scheduler(global_step)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            optimizer = self.optimizer(learning_rate)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # Add control dependencies to compute gradients and moving averages (batch normalization).
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([op_gradients] + update_ops):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            print(self._get_path('checkpoints'))
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, stddev=0.1, regularization=True):
        initial = tf.truncated_normal_initializer(0, stddev=stddev)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var) / stddev**2)
            self.regularizers_size.append(np.prod(shape))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=False):
        initial = tf.constant_initializer(0)
        # initial = tf.truncated_normal_initializer(0, stddev=1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            self.regularizers_size.append(np.prod(shape))
        tf.summary.histogram(var.op.name, var)
        return var


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        batch_norm: apply batch normalization after filtering (boolean vector)
        L: List of Graph Laplacians. Size M x M.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.
        statistics: layer which computes statistics from feature maps for the network to be invariant to translation and rotation.
            * None: no statistical layer (default)
            * 'mean': compute the mean of each feature map
            * 'var': compute the variance of each feature map
            * 'meanvar': compute the mean and variance of each feature map
            * 'histogram': compute a learned histogram of each feature map

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, L, F, K, p, batch_norm, M, 
                num_epochs, scheduler, optimizer, batch_norm_full=[],
                conv='chebyshev5', pool='max', activation='relu', statistics=None,
                regularization=0, dropout=1, batch_size=128, eval_frequency=200,
                dir_name='', profile=False, input_shape=None, debug=False):
        super(cgcnn, self).__init__()

        # Verify the consistency w.r.t. the number of layers.
        if not len(L) == len(F) == len(K) == len(p) == len(batch_norm):
            raise ValueError('Wrong specification of the convolutional layers: '
                             'parameters L, F, K, p, batch_norm, must have the same length.')
        if not np.all(np.array(p) >= 1):
            raise ValueError('Down-sampling factors p should be greater or equal to one.')
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        if not np.all(np.mod(p_log2, 1) == 0):
            raise ValueError('Down-sampling factors p should be powers of two.')
        if len(M) == 0 and p[-1] != 1:
            raise ValueError('Down-sampling should not be used in the last '
                             'layer if no fully connected layer follows.')

        # Keep the useful Laplacians only. May be zero.
        if len(L):
            M_0 = L[0].shape[0]
        else:
            M_0 = input_shape
        j = 0
        self.L = L

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        M_last = M_0
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if not (i == Ngconv-1 and len(M) == 0):  # No bias if it's a softmax.
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            if batch_norm[i]:
                print('    batch normalization')

        if Ngconv:
            M_last = L[-1].shape[0] * F[-1] // p[-1]

        if statistics is not None:
            print('  Statistical layer: {}'.format(statistics))
            if statistics is 'mean':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'var':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'meanvar':
                M_last = 2 * F[-1]
                print('    representation: 2 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'histogram':
                nbins = 20
                M_last = nbins * F[-1]
                print('    representation: {} * {} = {}'.format(nbins, F[-1], M_last))
                print('    weights: {} * {} = {}'.format(nbins, F[-1], M_last))
                print('    biases: {} * {} = {}'.format(nbins, F[-1], M_last))

        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            if i < Nfc - 1:  # No bias if it's a softmax.
                print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i]

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs = num_epochs
        self.scheduler, self.optimizer = scheduler, optimizer
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.batch_norm_full = batch_norm_full
        self.dir_name = dir_name
        self.filter = getattr(self, conv)
        self.pool = getattr(self, 'pool_' + pool)
        self.activation = getattr(tf.nn, activation)
        self.statistics = statistics
        self.profile, self.debug = profile, debug

        # Build the computational graph.
        self.build_graph(M_0)

        # show_all_variables()

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02*sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        W = self._weight_variable_cheby(K, Fin, Fout, regularization=True)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def _weight_variable_cheby(self, K, Fin, Fout, regularization=True):
        """Xavier like weight initializer for Chebychev coefficients."""
        stddev = 1 / np.sqrt(Fin * (K + 0.5) / 2)
        return self._weight_variable([Fin*K, Fout], stddev=stddev, regularization=regularization)

    def monomials(self, x, L, Fout, K):
        r"""Convolution on graph with monomials."""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02*sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to monomial basis.
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        for k in range(1, K):
            x1 = tf.sparse_tensor_dense_matmul(L, x0)  # M x Fin*N
            x = concat(x, x1)
            x0 = x1
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        W = self._weight_variable([Fin*K, Fout], regularization=True)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def bias(self, x):
        """Add one bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return x + b

    def pool_max(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def pool_average(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def learned_histogram(self, x, bins=20, initial_range=2):
        """A learned histogram layer.

        The center and width of each bin is optimized.
        One histogram is learned per feature map.
        """
        # Shape of x: #samples x #nodes x #features.
        n_features = int(x.get_shape()[2])
        centers = tf.linspace(-float(initial_range), initial_range, bins, name='range')
        centers = tf.expand_dims(centers, axis=1)
        centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
        centers = tf.Variable(
            tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
            name='centers',
            dtype=tf.float32)
        width = 4 * initial_range / bins  # 50% overlap between bins.
        widths = tf.get_variable(
            name='widths',
            shape=[1, 1, n_features, bins],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=width, dtype=tf.float32))
        x = tf.expand_dims(x, axis=3)
        # All are rank-4 tensors: samples, nodes, features, bins.
        widths = tf.abs(widths)
        dist = tf.abs(x - centers)
        hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1) * (bins/initial_range/4)
        return hist

    def batch_normalization(self, x, training, momentum=0.9):
        """Batch norm layer."""
        # Normalize over all but the last dimension, that is the features.
        return tf.layers.batch_normalization(x,
                                             axis=-1,
                                             momentum=momentum,
                                             epsilon=1e-5,
                                             center=False,  # Done by bias.
                                             scale=False,  # Done by filters.
                                             training=training)

    def fc(self, x, Mout, bias=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable_fc(int(Min), Mout, regularization=True)
        y = tf.matmul(x, W)
        if bias:
            y += self._bias_variable([Mout], regularization=False)
        return y

    def _weight_variable_fc(self, Min, Mout, regularization=True):
        """Xavier like weight initializer for fully connected layer."""
        stddev = 1 / np.sqrt(Min)
        return self._weight_variable([Min, Mout], stddev=stddev, regularization=regularization)

    def _inference(self, x, training):

        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                if i == len(self.p)-1 and len(self.M) == 0:
                    break  # That is a linear layer before the softmax.
                if self.batch_norm[i]:
                    x = self.batch_normalization(x, training)
                x = self.bias(x)
                x = self.activation(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])

        # Statistical layer (provides invariance to translation and rotation).
        with tf.variable_scope('stat'):
            n_samples, n_nodes, n_features = x.get_shape()
            if self.statistics is None:
                x = tf.reshape(x, [int(n_samples), int(n_nodes * n_features)])
            elif self.statistics is 'mean':
                x, _ = tf.nn.moments(x, axes=1)
            elif self.statistics is 'var':
                _, x = tf.nn.moments(x, axes=1)
            elif self.statistics is 'meanvar':
                mean, var = tf.nn.moments(x, axes=1)
                x = tf.concat([mean, var], axis=1)
            elif self.statistics is 'histogram':
                n_bins = 20
                x = self.learned_histogram(x, n_bins)
                x = tf.reshape(x, [int(n_samples), n_bins * int(n_features)])
            else:
                raise ValueError('Unknown statistical layer {}'.format(self.statistics))

        # Fully connected hidden layers.
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = self.activation(x)
                dropout = tf.cond(training, lambda: float(self.dropout), lambda: 1.0)
                x = tf.nn.dropout(x, dropout)
                if self.batch_norm_full[i]:
                    x = self.batch_normalization(x, training)
        # Logits linear layer, i.e. softmax without normalization.
        if len(self.M) != 0:
            with tf.variable_scope('logits'):
                x = self.fc(x, self.M[-1], bias=False)

        return x

    def get_filter_coeffs(self, layer, ind_in=None, ind_out=None):
        """Return the Chebyshev filter coefficients of a layer.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        K, Fout = self.K[layer-1], self.F[layer-1]
        trained_weights = self.get_var('conv{}/weights'.format(layer))  # Fin*K x Fout
        trained_weights = trained_weights.reshape((-1, K, Fout))
        if layer >= 2:
            Fin = self.F[layer-2]
            assert trained_weights.shape == (Fin, K, Fout)

        # Fin x K x Fout => K x Fout x Fin
        trained_weights = trained_weights.transpose([1, 2, 0])
        if ind_in:
            trained_weights = trained_weights[:, :, ind_in]
        if ind_out:
            trained_weights = trained_weights[:, ind_out, :]
        return trained_weights

    def plot_chebyshev_coeffs(self, layer, ind_in=None, ind_out=None,  ax=None, title='Chebyshev coefficients - layer {}'):
        """Plot the Chebyshev coefficients of a layer.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        title : figure title
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        trained_weights = self.get_filter_coeffs(layer, ind_in, ind_out)
        K, Fout, Fin = trained_weights.shape
        ax.plot(trained_weights.reshape((K, Fin*Fout)), '.')
        ax.set_title(title.format(layer))
        return ax


class cnn2d(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        batch_norm: apply batch normalization after filtering (boolean vector)
        L: List of Graph Laplacians. Size M x M.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.
        statistics: layer which computes statistics from feature maps for the network to be invariant to translation and rotation.
            * None: no statistical layer (default)
            * 'mean': compute the mean of each feature map
            * 'var': compute the variance of each feature map
            * 'meanvar': compute the mean and variance of each feature map
            * 'histogram': compute a learned histogram of each feature map

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, F, K, p, batch_norm, M, 
                num_epochs, scheduler, optimizer, batch_norm_full=[],
                pool='max', activation='relu', statistics=None,
                regularization=0, dropout=1, batch_size=128, eval_frequency=200,
                dir_name='', profile=False, input_shape=None, debug=False):
        super(cnn2d, self).__init__()

        # Verify the consistency w.r.t. the number of layers.
        if not len(F) == len(K) == len(p) == len(batch_norm):
            raise ValueError('Wrong specification of the convolutional layers: '
                             'parameters L, F, K, p, batch_norm, must have the same length.')
        if not np.all(np.array(p) >= 1):
            raise ValueError('Down-sampling factors p should be greater or equal to one.')
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        if not np.all(np.mod(p_log2, 1) == 0):
            raise ValueError('Down-sampling factors p should be powers of two.')
        if len(M) == 0 and p[-1] != 1:
            raise ValueError('Down-sampling should not be used in the last '
                             'layer if no fully connected layer follows.')
        j = 0

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: = {}'.format(input_shape))
        nx, ny = input_shape
        for i in range(Ngconv):
            nx = nx//p[i]
            ny = ny//p[i]
            print('  layer {0}: 2dconv{0}'.format(i+1))
            print('    representation: {0} x {1} x {2} = {3}'.format(nx, ny, F[i], nx*ny*F[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: {0} * {1} * {2} * {3} = {4}'.format(
                    K[i][0], K[i][1], F_last, F[i],  F_last*F[i]*K[i][0]*K[i][1]))
            if not (i == Ngconv-1 and len(M) == 0):  # No bias if it's a softmax.
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            if batch_norm[i]:
                print('    batch normalization')


        M_last = nx*ny * F[-1]

        if statistics is not None:
            print('  Statistical layer: {}'.format(statistics))
            if statistics is 'mean':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'var':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'meanvar':
                M_last = 2 * F[-1]
                print('    representation: 2 * {} = {}'.format(F[-1], M_last))
            elif statistics is 'histogram':
                nbins = 20
                M_last = nbins * F[-1]
                print('    representation: {} * {} = {}'.format(nbins, F[-1], M_last))
                print('    weights: {} * {} = {}'.format(nbins, F[-1], M_last))
                print('    biases: {} * {} = {}'.format(nbins, F[-1], M_last))

        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            if i < Nfc - 1:  # No bias if it's a softmax.
                print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i]

        # Store attributes and bind operations.
        self.F, self.K, self.p, self.M = F, K, p, M
        self.num_epochs = num_epochs
        self.scheduler, self.optimizer = scheduler, optimizer
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.batch_norm_full = batch_norm_full
        self.dir_name = dir_name
        self.input_shape = input_shape
        self.pool = getattr(self, 'pool_' + pool)
        self.activation = getattr(tf.nn, activation)
        self.statistics = statistics
        self.profile, self.debug = profile, debug

        # Build the computational graph.
        self.build_graph()

        
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_training: False}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def build_graph(self):
        """Build the computational graph of the model."""

        self.loadable_generator = LoadableGenerator()

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Make the dataset
            self.tf_train_dataset = tf.data.Dataset().from_generator(self.loadable_generator.iter, output_types=(tf.float32, tf.int32))
            self.tf_data_iterator = self.tf_train_dataset.prefetch(2).make_initializable_iterator()
            ph_data, ph_labels = self.tf_data_iterator.get_next()


            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder_with_default(ph_data, (self.batch_size, *self.input_shape), 'data')
                self.ph_labels = tf.placeholder_with_default(ph_labels, (self.batch_size), 'labels')
                self.ph_training = tf.placeholder(tf.bool, (), 'training')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_training)
            self.op_loss = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
            utils.show_all_variables()
        self.graph.finalize()
                  
    def conv2d(self, imgs, nf_out, shape=[5, 5], stride=2, scope="conv2d", regularization=True):
        '''Convolutional layer for square images'''

        if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
            stride = [stride, stride]

        weights_initializer = tf.contrib.layers.xavier_initializer()
#         const = tf.constant_initializer(0.0)

        with tf.variable_scope(scope):
            sh = [shape[0], shape[1], imgs.get_shape()[-1].value, nf_out]
            w = tf.get_variable('w', sh, initializer=weights_initializer)
            if regularization:
                self.regularizers.append(tf.nn.l2_loss(w))
                self.regularizers_size.append(np.prod(sh))
            conv = tf.nn.conv2d(
                imgs, w, strides=[1, *stride, 1], padding='SAME')

#             biases = _tf_variable('biases', [nf_out], initializer=const)
#             conv = tf.nn.bias_add(conv, biases)


            return conv

    def bias(self, x):
        """Add one bias per filter."""
        const = tf.constant_initializer(0.0)
        biases = tf.get_variable('biases', [x.shape[-1]], initializer=const)
        return tf.nn.bias_add(x, biases)

    def pool_max(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def pool_average(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def learned_histogram(self, x, bins=20, initial_range=2):
        """A learned histogram layer.

        The center and width of each bin is optimized.
        One histogram is learned per feature map.
        """
        # Shape of x: #samples x #nodes x #features.
        n_features = int(x.get_shape()[2])
        centers = tf.linspace(-float(initial_range), initial_range, bins, name='range')
        centers = tf.expand_dims(centers, axis=1)
        centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
        centers = tf.Variable(
            tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
            name='centers',
            dtype=tf.float32)
        width = 4 * initial_range / bins  # 50% overlap between bins.
        widths = tf.get_variable(
            name='widths',
            shape=[1, 1, n_features, bins],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=width, dtype=tf.float32))
        x = tf.expand_dims(x, axis=3)
        # All are rank-4 tensors: samples, nodes, features, bins.
        widths = tf.abs(widths)
        dist = tf.abs(x - centers)
        hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1) * (bins/initial_range/4)
        return hist

    def batch_normalization(self, x, training, momentum=0.9):
        """Batch norm layer."""
        # Normalize over all but the last dimension, that is the features.
        return tf.layers.batch_normalization(x,
                                             axis=-1,
                                             momentum=momentum,
                                             epsilon=1e-5,
                                             center=False,  # Done by bias.
                                             scale=False,  # Done by filters.
                                             training=training)

    def fc(self, x, Mout, bias=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable_fc(int(Min), Mout, regularization=True)
        y = tf.matmul(x, W)
        if bias:
            y += self._bias_variable([Mout], regularization=False)
        return y

    def _weight_variable_fc(self, Min, Mout, regularization=True):
        """Xavier like weight initializer for fully connected layer."""
        stddev = 1 / np.sqrt(Min)
        return self._weight_variable([Min, Mout], stddev=stddev, regularization=regularization)

    def _inference(self, x, training):
        x = tf.expand_dims(x,3)
        # Graph convolutional layers.
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.conv2d(x, self.F[i], self.K[i], self.p[i])
                if i == len(self.p)-1 and len(self.M) == 0:
                    break  # That is a linear layer before the softmax.
                if self.batch_norm[i]:
                    x = self.batch_normalization(x, training)
                x = self.bias(x)
                x = self.activation(x)
#                 with tf.name_scope('pooling'):
#                     x = self.pool(x, self.p[i])

        # Statistical layer (provides invariance to translation and rotation).
        with tf.variable_scope('stat'):
            n_samples, nx,ny, n_features = x.get_shape()
            n_nodes = nx*ny
            x = tf.reshape(x, (n_samples, n_nodes, n_features))
            if self.statistics is None:
                x = tf.reshape(x, [int(n_samples), int(n_nodes * n_features)])
            elif self.statistics is 'mean':
                x, _ = tf.nn.moments(x, axes=1)
            elif self.statistics is 'var':
                _, x = tf.nn.moments(x, axes=1)
            elif self.statistics is 'meanvar':
                mean, var = tf.nn.moments(x, axes=1)
                x = tf.concat([mean, var], axis=1)
            elif self.statistics is 'histogram':
                n_bins = 20
                x = self.learned_histogram(x, n_bins)
                x = tf.reshape(x, [int(n_samples), n_bins * int(n_features)])
            else:
                raise ValueError('Unknown statistical layer {}'.format(self.statistics))

        # Fully connected hidden layers.
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = self.activation(x)
                dropout = tf.cond(training, lambda: float(self.dropout), lambda: 1.0)
                x = tf.nn.dropout(x, dropout)
                if self.batch_norm_full[i]:
                    x = self.batch_normalization(x, training)
        # Logits linear layer, i.e. softmax without normalization.
        if len(self.M) != 0:
            with tf.variable_scope('logits'):
                x = self.fc(x, self.M[-1], bias=False)
        return x


class deepsphere(cgcnn):
    """
    Spherical convolutional neural network based on graph CNN

    The following are hyper-parameters of the spherical layers.
    They are lists, which length is equal to the number of gconv layers.
        nsides: NSIDE paramter of the healpix package
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        batch_norm: apply batch norm at the end of the filter (bool vector)

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.
        statistics: layer which computes statistics from feature maps for the network to be invariant to translation and rotation.
            * None: no statistical layer (default)
            * 'mean': compute the mean of each feature map
            * 'var': compute the variance of each feature map
            * 'meanvar': compute the mean and variance of each feature map
            * 'histogram': compute a learned histogram of each feature map

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, nsides, indexes=None, use_4=False, **kwargs):
        L, p = utils.build_laplacians(nsides, indexes=indexes, use_4=use_4)
        self.nsides = nsides
        self.pygsp_graphs = [None] * len(nsides)
        super(deepsphere, self).__init__(L=L, p=p, **kwargs)

    def get_gsp_filters(self, layer,  ind_in=None, ind_out=None):
        """Get the filter as a pygsp format

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from pygsp import filters

        trained_weights = self.get_filter_coeffs(layer, ind_in, ind_out)
        nside = self.nsides[layer-1]
        if self.pygsp_graphs[layer-1] is None:
            self.pygsp_graphs[layer-1] = utils.healpix_graph(nside=nside)
            self.pygsp_graphs[layer-1].estimate_lmax()
        return filters.Chebyshev(self.pygsp_graphs[layer-1], trained_weights)

    def plot_filters_spectral(self, layer,  ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter of a special layer in the spectral domain.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """
        import matplotlib.pyplot as plt

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)

        if ax is None:
            ax = plt.gca()
        filters.plot(sum=False, ax=ax, **kwargs)

        return ax

    def plot_filters_section(self, layer,  ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter section on the sphere

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_section(filters, order=self.K[layer-1], **kwargs)
        return fig

    def plot_filters_gnomonic(self, layer,  ind_in=None, ind_out=None, **kwargs):
        """Plot the filter localization on gnomonic view.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_gnomonic(filters, order=self.K[layer-1], **kwargs)

        return fig
