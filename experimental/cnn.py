import tensorflow as tf
import numpy as np
import yaml
from copy import deepcopy
from deepsphere import utils
import os, shutil, time, pickle
import sklearn

def build_index(level):
    values1 = np.arange(4).reshape([2,2])
    if level==1:
        values = values1
    else:
        values = np.zeros([2**level,2**level])
        lowerlevel = build_index(level-1)
        values += np.tile(lowerlevel,[2,2])  
        values += 4**(level-1)*np.repeat(np.repeat(values1,2**(level-1), axis=1), 2**(level-1), axis=0)
    return values

def _tf_variable(name, shape, initializer):
    """Create a tensorflow variable.

    Arguments
    --------
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    if True:  # with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def lrelu(x, leak=0.2, name="lrelu"):
    """Leak relu."""
    return tf.maximum(x, leak * x, name=name)

def batch_norm(x, train=True, epsilon=1e-5, momentum=0.8):
    return tf.contrib.layers.batch_norm(x,
                                        is_training=True,
                                        decay=momentum,
                                        epsilon=epsilon,
                                        center=False,  # Done by bias.
                                        scale=False,  # Done by filters.
                                        )

def conv2d(imgs, nf_out, shape=[5, 5], stride=2, scope="conv2d", summary=True):
    '''Convolutional layer for square images'''

    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride, stride]

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(scope):
        w = _tf_variable(
            'w', [shape[0], shape[1],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        conv = tf.nn.conv2d(
            imgs, w, strides=[1, *stride, 1], padding='SAME')

        biases = _tf_variable('biases', [nf_out], initializer=const)
        conv = tf.nn.bias_add(conv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv

def linear(input_, output_size, scope=None, summary=True):
    shape = input_.get_shape().as_list()

    weights_initializer = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(scope or "Linear"):
        matrix = _tf_variable(
            "Matrix", [shape[1], output_size],
            initializer=weights_initializer)
        bias = _tf_variable("bias", [output_size], initializer=const)
        if summary:
            tf.summary.histogram(
                "Matrix_sum", matrix, collections=["metrics"])
            tf.summary.histogram("Bias_sum", bias, collections=["metrics"])
        return tf.matmul(input_, matrix) + bias

def saferm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('Erase recursively directory: ' + path)
    if os.path.isfile(path):
        os.remove(path)
        print('Erase file: ' + path)

def arg_helper(params, d_param):
    for key in d_param.keys():
        params[key] = params.get(key, d_param[key])
        if type(params[key]) is dict:
            params[key] = arg_helper(params[key], d_param[key])
    return params


def rprint(msg, reuse=False):
    """Print message only if reuse is False.
    If a block is being resued, its description will not be re-printed.
    """
    if not reuse:
        print(msg)

class CNN(object):
    """Base Net abstract class."""
    def default_params(self):
        bn = False
        d_params = dict()
        d_params['full'] = [32, 2]
        d_params['nfilter'] = [16, 32, 32, 32]
        d_params['batch_norm'] = [bn, bn, bn, bn]
        d_params['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3]]
        d_params['stride'] = [2, 2, 2, 1]
        d_params['summary'] = True
        d_params['activation'] = tf.nn.relu #lrelu # leaky relu
        d_params['in_shape'] = [256, 256] # Shape of the image
        d_params['out_shape'] = [2] # Shape of the output (number of class)
        d_params['l2_reg'] = 0 # l2 regularization
        d_params['statistics'] = None # 'mean', 'var', 'meanvar'

        return d_params

    def __init__(self, params={}, name="BaseNet", debug_mode=False):
        self._debug_mode=debug_mode
        if self._debug_mode:
            print('User parameters...')
            print(yaml.dump(params))
        self._params = deepcopy(arg_helper(params, self.default_params()))
        if self._debug_mode:
            print('\nParameters used for the network...')
            print(yaml.dump(self._params))
        self._name = name
        self._outputs = None
        self._inputs = None
        self._loss = None
        self._build_net()
        self._add_summary()

    def _build_net(self):
        in_shape = self._params['in_shape']
        out_shape = self._params['out_shape']

        self.input = tf.placeholder(tf.float32, shape=[None, *in_shape, 1], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None, *out_shape], name='labels')
        is_training = tf.Variable(False, name='train_in', trainable=False)
        self.is_training = tf.placeholder_with_default(is_training, (), 'training')
        tf.summary.image('train/input', self.input, max_outputs=1, collections=["train"])

        print('  * Input shape : {}'.format(self.input.shape))
        self._logits = self.cnn(self.input , reuse=False)
        self._outputs = tf.nn.softmax(self._logits)
        self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits, labels=self.labels))
        print('  * Output shape : {}'.format(self._outputs.shape))
        
        if self._params['l2_reg']:
            vars   = tf.trainable_variables()
            self._lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * self._params['l2_reg']
            print('  * L2 regularization with weight: {}'.format(self._params['l2_reg']))
        else:
            self._lossL2 = 0
        self._loss = self._lossL2 + self._cross_entropy
        print('  * Loss shape {}'.format(self.loss.shape))

    def _add_summary(self):
        tf.summary.scalar('train/loss_reg_l2', self._lossL2, collections=["train"])
        tf.summary.scalar('train/loss_cross_entropy', self._cross_entropy, collections=["train"])
        tf.summary.scalar('train/loss', self._loss, collections=["train"])

    def cnn(self, x, reuse=False, scope="cnn"):
        params = self.params
        assert(len(params['stride']) ==
               len(params['nfilter']) ==
               len(params['batch_norm']))
        nconv = len(params['stride'])
        nfull = len(params['full'])

        bs = tf.shape(x)[0]

        with tf.variable_scope(scope, reuse=reuse):
            rprint('CNN architecture \n'+''.join(['-']*50), reuse)
            rprint('     The input is of size {}'.format(x.shape), reuse)
            for i in range(nconv):

                x = conv2d(x,
                         nf_out=params['nfilter'][i],
                         shape=params['shape'][i],
                         stride=params['stride'][i],
                         scope='{}_conv'.format(i),
                         summary=params['summary'])
                rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)

                if params['batch_norm'][i]:
                    x = batch_norm(x, train=self.is_training)
                    rprint('         Batch norm', reuse)
                rprint('         Size of the variables: {}'.format(x.shape), reuse)

                x = params['activation'](x)

            # Statistical layer (provides invariance to translation
            if params['statistics'] is not None:
                shape = x.shape.as_list()[1:]
                x = tf.reshape(x, [bs, shape[0]*shape[1], shape[2]])
                rprint('     Reshape to {}'.format(x.shape), reuse)

                if params['statistics']=='mean':
                    x, _ = tf.nn.moments(x, axes=1)
                elif params['statistics']=='var':
                    _, x = tf.nn.moments(x, axes=1)
                elif params['statistics']=='meanvar':
                    mean, var = tf.nn.moments(x, axes=1)
                    x = tf.concat([mean, var], axis=1)
                else:
                    raise ValueError('Unknown statistical layer {}'.format(self.statistics))

            x = tf.reshape(x, [bs, np.prod(x.shape.as_list()[1:])])
            rprint('     Reshape to {}'.format(x.shape), reuse)

            for i in range(nfull-1):
                x = linear(x,
                           params['full'][i],
                           '{}_full'.format(i+nconv),
                           summary=params['summary'])
                x = params['activation'](x)

                rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
                rprint('         Size of the variables: {}'.format(x.shape), reuse)
            if len(params['full']):
                x = linear(x, params['full'][-1], 'out', summary=params['summary'])
                # x = tf.sigmoid(x)
                rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
            rprint('     The output is of size {}'.format(x.shape), reuse)
            rprint(''.join(['-']*50)+'\n', reuse)
        return x

    def batch2dict(self, inputs):
        d = dict()
        sh = (inputs[0].shape[0], *self.params['in_shape'], 1)
        d['input'] = inputs[0].reshape(sh)
        d['labels'] =  np.eye(self.params['out_shape'][0])[inputs[1]]
        return d

    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss

    @property
    def outputs(self):
        return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def params(self):
        return self._params



class NNSystem(object):
    """A system to handle Neural Network"""
    def default_params(self):
        d_param = dict()
        d_param['optimization'] = dict()
        d_param['optimization']['learning_rate'] = 1e-4
        d_param['optimization']['batch_size'] = 8
        d_param['optimization']['epoch'] = 100
        d_param['optimization']['batch_size'] = 8

        d_param['net'] = dict()

        d_param['save_dir'] = './checkpoints/'
        d_param['summary_dir'] = './summaries/'
        d_param['summary_every'] = 200
        d_param['print_every'] = 100
        d_param['save_every'] = 10000
        return d_param

    def __init__(self, model, params={}, name=None, debug_mode=False):
        """Build the TF graph."""
        self._debug_mode=debug_mode
        if self._debug_mode:
            print('User parameters NNSystem...')
            print(yaml.dump(params))

        self._params = deepcopy(arg_helper(params, self.default_params()))
        if self._debug_mode:
            print('\nParameters used for the NNSystem..')
            print(yaml.dump(self._params))
        tf.reset_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default():
            if name:
                self._net = model(self.params['net'], name=name)
            else:
                self._net = model(self.params['net'])
            self._graph_init()
            self._params['net'] = deepcopy(self.net.params)
            self._name = self._net.name
            self._add_optimizer()
            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            self._summaries = tf.summary.merge(tf.get_collection("train"))
            self._opinit = tf.global_variables_initializer()
        self._graph.finalize()
        utils.show_all_variables()

    def _graph_init(self):
        pass

    def _add_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = self._params['optimization']['learning_rate']
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._optimize = optimizer.minimize(self._net.loss)
        tf.summary.scalar("training/loss", self._net.loss, collections=["train"])

    def _get_dict(self, index=None, **kwargs):
        """Return a dictionary with the argument for the architecture."""
        feed_dict = dict()
        for key, value in kwargs.items():
            if value is not None:
                    feed_dict[getattr(self._net, key)] = value
        if index:
            feed_dict = self._slice_feed_dict(feed_dict, index)
        return feed_dict

    def _slice_feed_dict(self, feed_dict, index):
        new_feed_dict = dict()
        for key, value in feed_dict.items():
            new_feed_dict[key] = value[index]
        return new_feed_dict

    def train(self, dataset, resume=False):

        n_data = dataset.N
        batch_size = self.params['optimization']['batch_size']
        self._counter = 1
        self._n_epoch = self.params['optimization']['epoch']
        self._total_iter = self._n_epoch * (n_data // batch_size) - 1
        self._n_batch = n_data // batch_size

        self._save_current_step = False

        # Create the save diretory if it does not exist
        os.makedirs(self._params['save_dir'], exist_ok=True)

        with tf.Session(graph=self._graph) as self._sess:
            if resume:
                print('Load weights in the network')
                self.load()
            else:
#                 self._sess.run(tf.global_variables_initializer())
                self._sess.run(self._opinit)
                
                saferm(self.params['summary_dir'])
                saferm(self.params['save_dir'])

            self._summary_writer = tf.summary.FileWriter(
                self._params['summary_dir'], self._sess.graph)
            try:
                self._epoch = 0
                self._time = dict()
                self._time['start_time'] = time.time()
                self._time['prev_iter_time'] = self._time['start_time']

                print('Start training')
                while self._epoch < self._n_epoch:
                    epoch_loss = 0.
                    diter = dataset.iter(batch_size)
                    for idx in range(dataset.N//batch_size):
                        batch = next(diter)
                        if resume:
                            self._counter = self.params['curr_counter']
                            resume = False
                        else:
                            self._params['curr_counter'] = self._counter
                        feed_dict = self._get_dict(**self._net.batch2dict(batch), is_training=True)
                        curr_loss = self._run_optimization(feed_dict, idx)
                        # epoch_loss += curr_loss

                        if np.mod(self._counter, self.params['print_every']) == 0:
                            # self._print_log(idx, curr_loss, epoch_loss/idx)
                            self._print_log(idx, curr_loss)

                        if np.mod(self._counter, self.params['summary_every']) == 0:
                            self._train_log(feed_dict)

                        if (np.mod(self._counter, self.params['save_every']) == 0) | self._save_current_step:
                            self._save(self._counter)
                            self._save_current_step = False
                        self._counter += 1
                    # epoch_loss /= self._n_batch
                    # print(" - Epoch {}, train loss: {:f}".format(self._epoch, epoch_loss))

                    self._epoch += 1
                print('Training done')
            except KeyboardInterrupt:
                pass
            self._save(self._counter)

    def _run_optimization(self, feed_dict, idx):
            if idx==0:
                self._epoch_loss = 0
            curr_loss = self._sess.run([self.net.loss, self._optimize], feed_dict)[0]
            self._epoch_loss += curr_loss
            return curr_loss

    def _print_log(self, idx, curr_loss):
        current_time = time.time()
        batch_size = self.params['optimization']['batch_size']
        print("    * Epoch: [{:2d}] [{:4d}/{:4d}] "
              "Counter:{:2d}\t"
              "({:4.1f} min\t"
              "{:4.3f} examples/sec\t"
              "{:4.2f} sec/batch)\t"
              "Batch loss:{:.8f}\t"
              "Mean loss:{:.8f}\t".format(
              self._epoch, 
              idx+1, 
              self._n_batch,
              self._counter,
              (current_time - self._time['start_time']) / 60,
              self._params['print_every'] * batch_size / (current_time - self._time['prev_iter_time']),
              (current_time - self._time['prev_iter_time']) / self._params['print_every'],
              curr_loss,
              self._epoch_loss/(idx+1)))
        self._time['prev_iter_time'] = current_time

    def _train_log(self, feed_dict):
        summary = self._sess.run(self._summaries, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)


    def _save(self, step):
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'])

        self._saver.save(
            self._sess,
            os.path.join(self.params['save_dir'], self._net.name),
            global_step=step)
        self._save_obj()
        print('Model saved!')

    def _save_obj(self):
        # Saving the objects:
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'], exist_ok=True)

        path_param = os.path.join(self.params['save_dir'], 'params.pkl')
        with open(path_param, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, sess=None, checkpoint=None):
        '''
        Given checkpoint, load the model.
        By default, load the latest model saved.
        '''
        if sess:
            self._sess = sess
        elif self._sess is None:
            raise ValueError("Session not available at the time of loading model!")

        if checkpoint:
            file_name = os.path.join(
                self.params['save_dir'],
                self.net.name+ '-' + str(checkpoint))
        else:
            file_name = None

        print(" [*] Reading checkpoints...")
        if file_name:
            self._saver.restore(self._sess, file_name)
            return True

        checkpoint_dir = self.params['save_dir']
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            return True
        print(" [*] No checkpoint found in {}".format(checkpoint_dir))
        return False


    def outputs(self, checkpoint=None, sess=None, is_training=False, **kwargs):
        close = False
        if sess is None:
            self._sess = tf.Session(graph=self._graph)
            close = True

            if self.load(checkpoint=checkpoint):
                print("Model loaded.")
            else:
                raise ValueError("Unable to load the model")
#         else:
#             self._sess.run(self._opinit)
        feed_dict = self._get_dict(is_training=is_training, **kwargs)
        ret = self._sess.run(self._net.outputs, feed_dict=feed_dict)
        if close:
            self._sess.close()
        return ret

    def loss(self, dataset, checkpoint=None):
        with tf.Session(graph=self._graph) as self._sess:

            if self.load(checkpoint=checkpoint):
                print("Model loaded.")
            else:
                raise ValueError("Unable to load the model")
            loss = 0
            batch_size = self._params['optimization']['batch_size']
            for idx, batch in enumerate(dataset.iter(batch_size)):
                feed_dict = self._get_dict(**self.net.batch2dict(batch), is_training=False)
                loss += self._sess.run(self.net.loss, feed_dict)
        return loss/idx
    @property
    def params(self):
        return self._params

    @property
    def net(self):
        return self._net
        

class ValidationNNSystem(NNSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _graph_init(self):
        self._validation_loss = tf.placeholder(tf.float32, name='validation_loss')
        self._validation_cross_entropy = tf.placeholder(tf.float32, name='validation_cross_entropy')
        self._validation_accuracy = tf.placeholder(tf.float32, name='validation_accuracy')
        self._validation_f1 = tf.placeholder(tf.float32, name='validation_f1')
        tf.summary.scalar("validation/loss", self._validation_loss, collections=["validation"])
        tf.summary.scalar('validation/loss_cross_entropy', self._validation_cross_entropy, collections=["validation"])
        tf.summary.scalar('validation/accuracy', self._validation_accuracy, collections=["validation"])
        tf.summary.scalar('validation/f1', self._validation_f1, collections=["validation"])

        self._summaries_validation = tf.summary.merge(tf.get_collection("validation"))


    def train(self, dataset_train, dataset_validation, resume=False):
        batch_size = self._params['optimization']['batch_size']
        self._validation_dataset = dataset_validation
        self._validation_dataset_iter = self._validation_dataset.iter(batch_size)
        super().train(dataset_train, resume=resume)

    def _train_log(self, feed_dict=dict()):
        super()._train_log(feed_dict)
        loss = 0
        batch_size = self._params['optimization']['batch_size']
        labels = []
        outputs = []
        nel = np.int(np.ceil(self._validation_dataset.N/batch_size))
        for idx in range(nel):
            batch = next(self._validation_dataset_iter)
            feed_dict2 = self._get_dict(**self._net.batch2dict(batch), is_training=False)
            loss, cross_entropy, label, output = self._sess.run([self._net.loss, self._net._cross_entropy, self.net.labels, self.net.outputs], feed_dict2)
            labels.append(label)
            outputs.append(output)
        loss = loss/nel
        
        labels = np.concatenate(labels)
        outputs = np.concatenate(outputs)

        predictions = np.argmax(outputs, axis=1)        
        labels = np.argmax(labels, axis=1)
        
        ncorrects = np.sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}'.format(
                accuracy, ncorrects, len(labels), f1)
        print(string)
        
        print("Validation loss: {}".format(loss))
        print("Validation cross entropy: {}".format(cross_entropy))
        feed_dict[self._validation_loss] = loss
        feed_dict[self._validation_cross_entropy] = cross_entropy
        feed_dict[self._validation_accuracy] = accuracy
        feed_dict[self._validation_f1] = f1
        summary = self._sess.run(self._summaries_validation, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)
        
        
class Healpix2CNN(object):
    def __init__(self, **kwargs):
        self.model = ValidationNNSystem(CNN, kwargs)
    def fit(self, *args, **kwargs):
        self.model.train(*args, **kwargs)
    def outputs(self, features, checkpoint=None):
        N = features.shape[0]
        sh = (N, *self.model.net.params['in_shape'], 1)
        feature_r = features.reshape(sh)
        batch_size = self.model.params['optimization']['batch_size']
        if N > batch_size:
            with tf.Session(graph=self.model._graph) as sess:
                if self.model.load(sess=sess, checkpoint=checkpoint):
                    print("Model loaded.")
                else:
                    raise ValueError("Unable to load the model")

                tmp = []
                for i in range(N//batch_size):
                    tmpdat = feature_r[i*batch_size:(i+1)*batch_size]
                    tmp.append(self.model.outputs(sess=sess, input=tmpdat))
                res = np.mod(N, batch_size)
                if res:
                    tmpdat = feature_r[-res:]
                    tmp.append(self.model.outputs(sess=sess, input=tmpdat))
                return np.concatenate(tmp)
        else:
            return self.model.outputs(checkpoint=checkpoint, input=feature_r)
    def predict(self, *args, **kwargs):
        outputs = self.outputs(*args, **kwargs)
        return np.argmax(outputs, axis=1)  