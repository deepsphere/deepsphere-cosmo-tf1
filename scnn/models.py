"""
This module defines the graph convolutional neural network.

Most of the code is based on https://github.com/mdeff/cnn_graph/.
"""
from __future__ import division

import os
import time
import collections
import shutil

import numpy as np
from scipy import sparse
import sklearn
import tensorflow as tf
from builtins import range

from . import utils

def show_all_variables():
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# Python 2 compatibility.
if hasattr(time, 'process_time'):
    process_time = time.process_time
else:
    import warnings
    warnings.warn('The CPU time is not working with Python 2.')
    def process_time():
        return np.nan


class base_model(object):
    """Common methods for all models."""

    def __init__(self):
        self.regularizers = []

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
        t_process, t_wall = process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(process_time()-t_process, time.time()-t_wall)
        return string, accuracy, f1, loss

    def fit(self, train_dataset, val_dataset):
        t_process, t_wall = process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_dataset.N / self.batch_size)
        train_iter = train_dataset.iter(self.batch_size)
        val_data, val_labels = val_dataset.get_all_data()
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_dataset.N))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = next(train_iter)
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_training: True}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_dataset.N
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

#                 _, accuracy2, f12, loss2 = self.evaluate(train_data, train_labels, sess)
#                 print('  Train data: accuracy {:.2e}, f1 {:.2e}, loss {:.2e}' .format(accuracy2, f12, loss2))

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.

    def build_graph(self, M_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_training = tf.placeholder(tf.bool, (), 'training')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_training)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum, self.adam)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

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
                regularization *= tf.add_n(self.regularizers)/len(self.regularizers)
            loss = cross_entropy + regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9, adam=False, beta2=0.999):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
                tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if adam:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=momentum, beta2=beta2)
            else:
                if momentum == 0:
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                else:
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
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

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
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
        batch_norm: apply batch norm at the end of the filter (bool vector)
        L: List of Graph Laplacians. Size M x M.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
        statistical_layer: layer which computes statistics from feature maps for the network to be invariant to translation and rotation.
            * None: no statistical layer (default)
            * 'mean': compute the mean of each feature map
            * 'var': compute the variance of each feature map
            * 'meanvar': compute the mean and variance of each feature map
            * 'histogram': compute a learned histogram of each feature map

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.
        adam:          Use adam optimizer.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, L, F, K, p, batch_norm, M, conv='chebyshev5', brelu='b1relu', pool='mpool1', statistical_layer=None,
                num_epochs=20, adam=True, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name=''):
        super(cgcnn, self).__init__()

        # Verify the consistency w.r.t. the number of layers.
        assert len(L) == len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.

        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
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
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))

        if Ngconv:
            M_last = L[-1].shape[0] * F[-1] // p[-1]

        if statistical_layer is not None:
            print('  Statistical layer: {}'.format(statistical_layer))
            if statistical_layer is 'mean':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistical_layer is 'var':
                M_last = F[-1]
                print('    representation: 1 * {} = {}'.format(F[-1], M_last))
            elif statistical_layer is 'meanvar':
                M_last = 2 * F[-1]
                print('    representation: 2 * {} = {}'.format(F[-1], M_last))
            elif statistical_layer is 'histogram':
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
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i]

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.adam = adam
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.dir_name = dir_name
        self.filter = getattr(self, conv)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.statistical_layer = statistical_layer

        # Build the computational graph.
        self.build_graph(M_0)

        show_all_variables()

    def chebyshev5(self, x, L, Fout, K):
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
        W = self._weight_variable([Fin*K, Fout], regularization=False)
#         W = tf.Variable(initial_value=np.ones([Fin*K, Fout]), trainable=False, dtype=tf.float32)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

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
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b1lrelu(self, x):
        """Bias and Leak ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return self.lrelu(x + b)

    def lrelu(self, x, leak=0.2):
        ''' Leak relu '''
        return tf.maximum(x, leak*x)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2lrelu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return self.lrelu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def histogram_layer(self, x, nbins=20, initial_range=2):

        # Variables: center and width of bins.
        wi = tf.linspace(-float(initial_range), initial_range, nbins, name='range')
        mu = tf.Variable(
            tf.reshape(wi, shape=[1, 1, nbins]),
            name='mu',
            dtype=tf.float32)
        w = tf.get_variable(
            name='w',
            shape=[1, 1, nbins],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=1, dtype=tf.float32))
        # reshape 2d
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        # Add dimension for the computations
        x = tf.expand_dims(x, axis=2)
        hist = tf.reduce_mean(tf.nn.relu(1-tf.abs(x-mu)*tf.abs(w)), axis=1)
        return hist

    def batch_normalization(self, x, training, epsilon=1e-5, decay=0.9):
        return tf.contrib.layers.batch_norm(x,
                                            decay=decay,
                                            updates_collections=None,
                                            epsilon=epsilon,
                                            scale=True,
                                            is_training=training)

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, training):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('batch_norm'):
                    if self.batch_norm[i]:
                        x = self.batch_normalization(x, training)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])

        if self.statistical_layer is None:
            N, M, F = x.get_shape()  # #samples x #nodes x #features
            x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        elif self.statistical_layer is 'mean':
            x, _ = tf.nn.moments(x, axes=1)
        elif self.statistical_layer is 'var':
            _, x = tf.nn.moments(x, axes=1)
        elif self.statistical_layer is 'meanvar':
            mean, var = tf.nn.moments(x, axes=1)
            x = tf.concat([mean, var], axis=1)
        elif self.statistical_layer is 'histogram':
            x = self.histogram_layer(x)
        else:
            raise ValueError('Unknown statistical_layer type')

        # Fully connected hidden layers.
        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                dropout = tf.cond(training, lambda: float(self.dropout), lambda: 1.0)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x

    def get_filter_coeffs(self, layer, ind_in=None, ind_out=None):
        """Return the Chebyshev filter coefficients of a layer.

        Arguments
        ---------
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

        Arguments
        ---------
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


class scnn(cgcnn):
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
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.
        adam:          Use adam optimizer.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, nsides, F, K, batch_norm, M, indexes=None, use_4=False, **kwargs):

        L, p = utils.build_laplacians(nsides, indexes=indexes, use_4=use_4)
        self.nsides = nsides
        self.pygsp_graphs = [None]*len(nsides)
        super(scnn, self).__init__(L, F, K, p, batch_norm, M, **kwargs)


    def get_gsp_filters(self, layer,  ind_in=None, ind_out=None):
        """Get the filter as a pygsp format

        Arguments
        ---------
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

        Arguments
        ---------
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

        Arguments
        ---------
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
        """Plot the filter section on the sphere

        Arguments
        ---------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_gnomonic(filters, order=self.K[layer-1], **kwargs)

        return fig
