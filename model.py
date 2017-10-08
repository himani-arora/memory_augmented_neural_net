import tensorflow as tf
import numpy as np


class NTMCopyModel():
    def __init__(self, args, seq_length, reuse=False):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[args.batch_size, seq_length, args.vector_dim])
        self.y = self.x
        eof = np.zeros([args.batch_size, args.vector_dim + 1])
        eof[:, args.vector_dim] = np.ones([args.batch_size])
        eof = tf.constant(eof, dtype=tf.float32)
        zero = tf.constant(np.zeros([args.batch_size, args.vector_dim + 1]), dtype=tf.float32)

        if args.model == 'LSTM':
            # single_cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size)
            # cannot use [single_cell] * 3 in tensorflow 1.2
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size, reuse=reuse)
            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'NTM':
            import ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim, 1, 1,
                                    addressing_mode='content_and_location',
                                    reuse=reuse,
                                    output_dim=args.vector_dim)

        state = cell.zero_state(args.batch_size, tf.float32)
        self.state_list = [state]
        for t in range(seq_length):
            output, state = cell(tf.concat([self.x[:, t, :], np.zeros([args.batch_size, 1])], axis=1), state)
            self.state_list.append(state)
        output, state = cell(eof, state)
        self.state_list.append(state)

        self.o = []
        for t in range(seq_length):
            output, state = cell(zero, state)
            self.o.append(output[:, 0:args.vector_dim])
            self.state_list.append(state)
        self.o = tf.sigmoid(tf.transpose(self.o, perm=[1, 0, 2]))

        # self.copy_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.o), reduction_indices=[1, 2]))
        eps = 1e-8
        self.copy_loss = -tf.reduce_mean(   # cross entropy function
            self.y * tf.log(self.o + eps) + (1 - self.y) * tf.log(1 - self.o + eps)
        )
        with tf.variable_scope('optimizer', reuse=reuse):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, momentum=0.9, decay=0.95)
            gvs = self.optimizer.compute_gradients(self.copy_loss)
            capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
        self.copy_loss_summary = tf.summary.scalar('copy_loss_%d' % seq_length, self.copy_loss)
        # self.merged_summary = tf.summary.merge(self.copy_loss_summary)


class NTMOneShotLearningModel():
    def __init__(self, args):
        if args.label_type == 'one_hot':
            args.output_dim = args.n_classes
        elif args.label_type == 'five_hot':
            args.output_dim = 25

        self.x_image = tf.placeholder(dtype=tf.float32,
                                      shape=[args.batch_size, args.seq_length, args.image_width * args.image_height* args.image_channel])
        self.x_label = tf.placeholder(dtype=tf.float32,
                                      shape=[args.batch_size, args.seq_length, args.output_dim])
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[args.batch_size, args.seq_length, args.output_dim])
        self.is_training=tf.placeholder(tf.bool)

        if args.embedder == 'CNN':
            self.embedder = LeNet(args.image_height, args.image_width, args.image_channel, args.embed_dim)

        if args.model == 'LSTM':
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'NTM':
            import ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    read_head_num=args.read_head_num,
                                    write_head_num=args.write_head_num,
                                    addressing_mode='content_and_location',
                                    output_dim=args.output_dim)
        elif args.model == 'MANN':
            import mann_cell as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    head_num=args.read_head_num)
        elif args.model == 'MANN2':
            import mann_cell_2 as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    head_num=args.read_head_num)

        state = cell.zero_state(args.batch_size, tf.float32)
        self.state_list = [state]   # For debugging
        self.o = []
        for t in range(args.seq_length):
            
            if args.embedder == 'CNN':
                self.x_batch=self.embedder.core_builder(self.x_image[:, t, :],self.is_training)
            else:
                self.x_batch=self.x_image[:, t, :]

            print('X batch input shape: %s'%(self.x_batch.get_shape()))

            output, state = cell(tf.concat([self.x_batch, self.x_label[:, t, :]], axis=1), state)
            print('MANN cell output shape: %s'%(output.get_shape()))
            # output, state = cell(self.y[:, t, :], state)
            with tf.variable_scope("o2o", reuse=(t > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)
            print('FC output shape: %s'%(output.get_shape()))
            if args.label_type == 'one_hot':
                output = tf.nn.softmax(output, dim=1)
            elif args.label_type == 'five_hot':
                output = tf.stack([tf.nn.softmax(o) for o in tf.split(output, 5, axis=1)], axis=1)
            self.o.append(output)
            self.state_list.append(state)
        self.o = tf.stack(self.o, axis=1)
        self.state_list.append(state)

        eps = 1e-8
        if args.label_type == 'one_hot':
            self.learning_loss = -tf.reduce_mean(  # cross entropy function
                tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1, 2])
            )
        elif args.label_type == 'five_hot':
            self.learning_loss = -tf.reduce_mean(  # cross entropy function
                tf.reduce_sum(tf.stack(tf.split(self.y, 5, axis=2), axis=2) * tf.log(self.o + eps), axis=[1, 2, 3])
            )
        self.o = tf.reshape(self.o, shape=[args.batch_size, args.seq_length, -1])
        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(
            #     learning_rate=args.learning_rate, momentum=0.9, decay=0.95
            # )
            # gvs = self.optimizer.compute_gradients(self.learning_loss)
            # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            # self.train_op = self.optimizer.apply_gradients(gvs)
            self.train_op = self.optimizer.minimize(self.learning_loss)


class LeNet(object):
  """Standard CNN architecture."""

  def __init__(self,image_height,image_width, num_channels, hidden_dim):
    self.image_height= image_height
    self.image_width= image_width
    self.num_channels = num_channels
    self.hidden_dim = hidden_dim
    self.matrix_init = tf.truncated_normal_initializer(stddev=0.1)
    self.vector_init = tf.constant_initializer(0.0)

    ch1 = 64  # number of channels in 1st layer
    ch2 = 64  # number of channels in 2nd layer
    ch3 = 64  # number of channels in 3rd layer
    ch4 = 64  # number of channels in 4th layer

    with tf.variable_scope('embedder', reuse=False):

        self.conv1_weights = tf.get_variable('conv1_w',[3, 3, self.num_channels, ch1],
            initializer=self.matrix_init)
        self.conv1_biases = tf.get_variable('conv1_b', [ch1],initializer=self.vector_init)

        self.conv2_weights = tf.get_variable('conv2_w', [3, 3, ch1, ch2],
            initializer=self.matrix_init)
        self.conv2_biases = tf.get_variable('conv2_b', [ch2],initializer=self.vector_init)

        self.conv3_weights = tf.get_variable('conv3_w', [3, 3, ch2, ch3],
            initializer=self.matrix_init)
        self.conv3_biases = tf.get_variable('conv3_b', [ch3],initializer=self.vector_init)

        self.conv4_weights = tf.get_variable('conv4_w', [3, 3, ch3, ch4],
            initializer=self.matrix_init)
        self.conv4_biases = tf.get_variable('conv4_b', [ch4],initializer=self.vector_init)
    
        # fully connected
        self.fc1_weights = tf.get_variable(
            'fc1_w', [self.image_width // 16 * self.image_height // 16 * ch4,self.hidden_dim], 
            initializer=self.matrix_init)
        self.fc1_biases = tf.get_variable('fc1_b', [self.hidden_dim],initializer=self.vector_init)

  def core_builder(self, x, phase):
    """Embeds x using standard CNN architecture.

    Args:
      x: Batch of images as a 2-d Tensor [batch_size, -1].

    Returns:
      A 2-d Tensor [batch_size, hidden_dim] of embedded images.
    """

    # define model
    x = tf.reshape(x,
                   [-1, self.image_height, self.image_width, self.num_channels])
    batch_size = tf.shape(x)[0]

    conv1 = tf.nn.conv2d(x, self.conv1_weights,strides=[1, 1, 1, 1], padding='SAME')
    #conv1=tf.contrib.layers.batch_norm(conv1, center=True, is_training=phase)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.conv2d(pool1, self.conv2_weights,strides=[1, 1, 1, 1], padding='SAME')
    #conv2=tf.contrib.layers.batch_norm(conv2, center=True, is_training=phase)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.conv2d(pool2, self.conv3_weights,strides=[1, 1, 1, 1], padding='SAME')
    #conv3=tf.contrib.layers.batch_norm(conv3, center=True, is_training=phase)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_biases))
    pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    conv4 = tf.nn.conv2d(pool3, self.conv4_weights,strides=[1, 1, 1, 1], padding='SAME')
    #conv4=tf.contrib.layers.batch_norm(conv4, center=True, is_training=phase)
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, self.conv4_biases))
    pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    reshape = tf.reshape(pool4, [batch_size, -1])
    hidden = tf.matmul(reshape, self.fc1_weights) + self.fc1_biases

    return hidden
