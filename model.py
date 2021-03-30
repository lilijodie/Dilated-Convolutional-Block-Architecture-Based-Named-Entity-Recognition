import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import tensorflow.contrib.slim as slim
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def __init_weights(self,shape, name):
        '''
        Initializer of convolutional kernel.

        Forming a convolutional kernel with initial weights by tf.truncated_normal method.

        :param shape: A `List` representing the shape of the convolutional kernel.
        :param name: A `Str` representing the name of the convolutional kernel.
        :return: A `tf.Variable` representing the convolutional kernel with initial weights.
        '''
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1),
                           name=name)  # the default value of stddev is set as 0.1

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        # self.dbcn_op()
        # self.odcn_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, 100], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)

    def dbcn_op(self):
        # The following codes initialize the weights of convolutional kernels in the model.
        w_residual = self.__init_weights(shape=[99, 50, 1, 1], name="W_residual")
        # the residual convolutional kernel with the shape of (width,height,channel,out depth) where the channel is 1.
        w_dilatedblock_1 = self.__init_weights(shape=[41, 50, 1, 6], name="W_dilatedblock1")
        # the kernel of the first dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.
        w_dilatedblock_2 = self.__init_weights(shape=[7, 50, 6, 1], name="W_dilatedblock2")
        # the kernel of the second dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.

        # The following codes reshape and unstack the input in order to let the input adapt the operation of column-wise dilated convolutional layer.
        with tf.name_scope("inputs"):
            inputs_reshape = tf.expand_dims(self.word_embeddings,
                                            -1)  # reshape the inputs (expand the dimension of inputs from 3-D to 4-D (Batch,GroupNum,Embedding Size)=>(Batch,GroupNum,Embedding Size,1)).
            inputs_unstack = tf.unstack(inputs_reshape,
                                        axis=2)  # unstack the inputs on the 3rd axis--Embedding Size, the shape of inputs will change to Embedding Size*(Batch,GroupNum,1).

        # The following codes send the inputs into residual convolutional layers and perform related operations.
        with tf.name_scope("residual_convolution"):
            convs = []  # for collecting the residual convolution results.
            w_unstack = tf.unstack(w_residual,
                                   axis=1)  # unstack the residual convolutional kernel for column-wise convolution.
            # column-wise convolution
            for i in range(len(inputs_unstack)):
                conv = tf.nn.convolution(input=inputs_unstack[i], filter=w_unstack[i], padding="VALID")
                convs.append(conv)
            convres = tf.stack(convs, axis=2)  # for stacking the residual convolution results (on the 3rd axis).
            print("residual convolution:" + str(convres))

        # The following codes send the inputs into first dilated block and perform related operations.
        with tf.name_scope("dilated_block_1"):
            convs1 = []  # for collecting the first dilated block results.
            w1_unstack = tf.unstack(w_dilatedblock_1,
                                    axis=1)  # unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
            # column-wise dilated convolution, batch normalization and activation
            for i in range(len(inputs_unstack)):
                conv1 = tf.nn.convolution(input=inputs_unstack[i], filter=w1_unstack[i], padding="VALID",
                                          dilation_rate=[2])
                bn1 = tf.layers.batch_normalization(conv1, training=False)
                ac1 = tf.nn.relu(bn1)
                convs1.append(ac1)
            convres1 = tf.stack(convs1, axis=2)  # for stacking the first dilated block results (on the 3rd axis).
            print("dilated block 1:" + str(convres1))

        # The following codes send the first dilated block results into second dilated block and perform related operations.
        with tf.name_scope("dilated_block_2"):
            convs2 = []  # for collecting the second dilated block results.
            convres1_unstack = tf.unstack(convres1,
                                          axis=2)  # unstack the results of the first dilated block for column-wise dilated convolution.
            w2_unstack = tf.unstack(w_dilatedblock_2,
                                    axis=1)  # unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
            # column-wise dilated convolution, batch normalization and activation
            for i in range(len(convres1_unstack)):
                conv2 = tf.nn.convolution(input=convres1_unstack[i], filter=w2_unstack[i], padding="VALID",
                                          dilation_rate=[3])
                bn2 = tf.layers.batch_normalization(conv2, training=False)
                ac2 = tf.nn.relu(bn2)
                convs2.append(ac2)
            convres2 = tf.stack(convs2, axis=2)  # for stacking the second dilated block results (on the 3rd axis).
            print("dilated block 2:" + str(convres2))

        # The following codes concatenate the results of second dilated block and the results residual convolution and perform other operations.
        with tf.name_scope("concat_pool_flat_output"):
            concatres = tf.concat([convres, convres2], axis=1)  # concatenation.
            print("concat:" + str(concatres))
            poolres = tf.nn.max_pool(value=concatres, ksize=[1, 1, 1, 1],
                                     strides=[1, 1, 1, 1], padding="VALID")  # maxpooling.
            print("pooling:" + str(poolres))
            flatres = slim.flatten(poolres)  # flat.
            print("flat:" + str(flatres))
        with tf.name_scope("concat_dbcn_word"):
            word_embedding_unstack=tf.unstack(self.word_embeddings,axis=1)
            finalresult=[]
            for i in range(len(word_embedding_unstack)):
                finalresult.append(tf.concat([word_embedding_unstack[i],flatres],axis=1))
            self.word_embeddings=tf.stack(finalresult, axis=1)
            print("word_embedding:"+str(self.word_embeddings))

    def odcn_op(self):
        # The following codes initialize the weights of convolutional kernels in the model.
        w_dilatedblock_1 = self.__init_weights(shape=[41, 50, 1, 6], name="W_dilatedblock1")
        # the kernel of the first dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.
        w_dilatedblock_2 = self.__init_weights(shape=[7, 50, 6, 1], name="W_dilatedblock2")
        # the kernel of the second dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.

        # The following codes reshape and unstack the input in order to let the input adapt the operation of column-wise dilated convolutional layer.
        with tf.name_scope("inputs"):
            inputs_reshape = tf.expand_dims(self.word_embeddings,
                                            -1)  # reshape the inputs (expand the dimension of inputs from 3-D to 4-D (Batch,GroupNum,Embedding Size)=>(Batch,GroupNum,Embedding Size,1)).
            inputs_unstack = tf.unstack(inputs_reshape,
                                        axis=2)  # unstack the inputs on the 3rd axis--Embedding Size, the shape of inputs will change to Embedding Size*(Batch,GroupNum,1).


        # The following codes send the inputs into first dilated block and perform related operations.
        with tf.name_scope("dilated_block_1"):
            convs1 = []  # for collecting the first dilated block results.
            w1_unstack = tf.unstack(w_dilatedblock_1,
                                    axis=1)  # unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
            # column-wise dilated convolution, batch normalization and activation
            for i in range(len(inputs_unstack)):
                conv1 = tf.nn.convolution(input=inputs_unstack[i], filter=w1_unstack[i], padding="VALID",
                                          dilation_rate=[2])
                bn1 = tf.layers.batch_normalization(conv1, training=False)
                ac1 = tf.nn.relu(bn1)
                convs1.append(ac1)
            convres1 = tf.stack(convs1, axis=2)  # for stacking the first dilated block results (on the 3rd axis).
            print("dilated block 1:" + str(convres1))

        # The following codes send the first dilated block results into second dilated block and perform related operations.
        with tf.name_scope("dilated_block_2"):
            convs2 = []  # for collecting the second dilated block results.
            convres1_unstack = tf.unstack(convres1,
                                          axis=2)  # unstack the results of the first dilated block for column-wise dilated convolution.
            w2_unstack = tf.unstack(w_dilatedblock_2,
                                    axis=1)  # unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
            # column-wise dilated convolution, batch normalization and activation
            for i in range(len(convres1_unstack)):
                conv2 = tf.nn.convolution(input=convres1_unstack[i], filter=w2_unstack[i], padding="VALID",
                                          dilation_rate=[3])
                bn2 = tf.layers.batch_normalization(conv2, training=False)
                ac2 = tf.nn.relu(bn2)
                convs2.append(ac2)
            convres2 = tf.stack(convs2, axis=2)  # for stacking the second dilated block results (on the 3rd axis).
            print("dilated block 2:" + str(convres2))

        # The following codes concatenate the results of second dilated block and the results residual convolution and perform other operations.
        with tf.name_scope("concat_pool_flat_output"):
            poolres = tf.nn.max_pool(value=convres2, ksize=[1, 1, 1, 1],
                                     strides=[1, 1, 1, 1], padding="VALID")  # maxpooling.
            print("pooling:" + str(poolres))
            flatres = slim.flatten(poolres)  # flat.
            print("flat:" + str(flatres))
        with tf.name_scope("concat_odcn_word"):
            word_embedding_unstack=tf.unstack(self.word_embeddings,axis=1)
            finalresult=[]
            for i in range(len(word_embedding_unstack)):
                finalresult.append(tf.concat([word_embedding_unstack[i],flatres],axis=1))
            self.word_embeddings=tf.stack(finalresult, axis=1)
            print("word_embedding:"+str(self.word_embeddings))

    # def odcn_op(self):
    #     # The following codes initialize the weights of convolutional kernels in the model.
    #     w_dilatedblock_1 = self.__init_weights(shape=[10, 50, 1, 1], name="W_dilatedblock1")
    #     # the kernel of the first dilated convolutional layer with the shape of (width,height,channel,out depth) where the channel is 1.
    #
    #     # The following codes reshape and unstack the input in order to let the input adapt the operation of column-wise dilated convolutional layer.
    #     with tf.name_scope("inputs"):
    #         inputs_reshape = tf.expand_dims(self.word_embeddings,
    #                                         -1)  # reshape the inputs (expand the dimension of inputs from 3-D to 4-D (Batch,GroupNum,Embedding Size)=>(Batch,GroupNum,Embedding Size,1)).
    #         inputs_unstack = tf.unstack(inputs_reshape,
    #                                     axis=2)  # unstack the inputs on the 3rd axis--Embedding Size, the shape of inputs will change to Embedding Size*(Batch,GroupNum,1).
    #
    #     # The following codes send the inputs into first dilated block and perform related operations.
    #     with tf.name_scope("dilated_block_1"):
    #         convs1 = []  # for collecting the first dilated block results.
    #         w1_unstack = tf.unstack(w_dilatedblock_1,
    #                                 axis=1)  # unstack the kernel of the first dilated convolutional layer for column-wise dilated convolution.
    #         # column-wise dilated convolution, batch normalization and activation
    #         for i in range(len(inputs_unstack)):
    #             conv1 = tf.nn.convolution(input=inputs_unstack[i], filter=w1_unstack[i], padding="VALID",
    #                                       dilation_rate=[8])
    #             bn1 = tf.layers.batch_normalization(conv1, training=False)
    #             ac1 = tf.nn.relu(bn1)
    #             convs1.append(ac1)
    #         convres1 = tf.stack(convs1, axis=2)  # for stacking the first dilated block results (on the 3rd axis).
    #         print("dilated block 1:" + str(convres1))
    #
    #     # The following codes concatenate the results of second dilated block and the results residual convolution and perform other operations.
    #     with tf.name_scope("concat_pool_flat_output"):
    #         poolres = tf.nn.max_pool(value=convres1, ksize=[1, 1, 1, 1],
    #                                  strides=[1, 1, 1, 1], padding="VALID")  # maxpooling.
    #         print("pooling:" + str(poolres))
    #         flatres = slim.flatten(poolres)  # flat.
    #         print("flat:" + str(flatres))
    #     with tf.name_scope("concat_odcn_word"):
    #         word_embedding_unstack = tf.unstack(self.word_embeddings, axis=1)
    #         finalresult = []
    #         for i in range(len(word_embedding_unstack)):
    #             finalresult.append(tf.concat([word_embedding_unstack[i], flatres], axis=1))
    #         self.word_embeddings = tf.stack(finalresult, axis=1)
    #         print("word_embedding:" + str(self.word_embeddings))


    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(sent))
                print(label_)
                print(len(label_))
                print(tag)
                print(len(tag))
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join("C:/Users/yuanyue/Desktop/idcnn_softmax/", 'label_' + epoch_num)
        metric_path = os.path.join("C:/Users/yuanyue/Desktop/idcnn_softmax/", 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

