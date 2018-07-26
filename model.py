# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob, window_context


class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.word_dim = config["word_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]
        self.pos_dim = config["pos_dim"]
        self.windows = config["windows"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_words = config["num_words"]
        self.num_segs = 4
        self.num_poss = 40

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.word_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="WordInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        self.pos_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="PosInputs")
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
	###########################
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # embeddings for chinese character and segmentation representation
        self.embedding = self.embedding_layer(self.char_inputs, self.word_inputs, self.seg_inputs, self.pos_inputs, config)


        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(self.embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, word_inputs, seg_inputs, pos_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """
        #print('XXXXXXXXXXXXXXXXXXXXXXXXX')char_dim
        #print(len(char_inputs))
        #print("chars len: %i / words len: %i / segs len: %i / tags len: %i." % (
        #len(chars), len(words), len(segs), len(tags)))
        windows=config["windows"]
        '''
        windows=config["windows"]
        padding=0
        if windows>0 :
        	padding=windows//2
        	for i in range(padding):	
        		char_inputs.insert(0, 0)
        		word_inputs.insert(0, 0)
        		char_inputs.append(0)
        		word_inputs.append(0)
        '''
        embedding = []
        embedding_char = []
        embedding_word = []
        #embedding_seg = []
        #'''
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
        #'''
        #'''
        with tf.variable_scope("word_embedding" if not name else name), tf.device('/cpu:0'):
            self.word_lookup = tf.get_variable(
                    name="word_embedding",
                    shape=[self.num_words, self.word_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.word_lookup, word_inputs))
        #'''
        #if windows>0:
        	#embedding=window_context(embedding, windows)

        #embedding=np.array(embedding_char+embedding_word)*0.5
        #embedding=embedding.tolist()
        #
        if config["seg_dim"]:
            with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                    name="seg_embedding",
                    shape=[self.num_segs, self.seg_dim],
                    initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))

        if config["pos_dim"]:
            with tf.variable_scope("pos_embedding"), tf.device('/cpu:0'):
                    self.pos_lookup = tf.get_variable(
                    name="pos_embedding",
                    shape=[self.num_poss, self.pos_dim],
                    initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.pos_lookup, pos_inputs))


        embed = tf.concat(embedding, axis=-1)
            #print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(embed)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, words, segs, poss, tags = batch
        #print("chars len: %i / words len: %i / segs len: %i / poss len: %i / tags len: %i." % (
        #len(chars), len(words), len(segs), len(poss), len(tags)))
        feed_dict = {
            #self.char_inputs: contextwin(np.asarray(chars), 5),
            #self.word_inputs: contextwin(np.asarray(words), 5),
            self.char_inputs: np.asarray(chars),
            self.word_inputs: np.asarray(words),
            self.seg_inputs: np.asarray(segs),
            self.pos_inputs: np.asarray(poss),
            self.dropout: 1.0,
        }
        #print(feed_dict)
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        #pad_id_char=char_to_id["<PAD>"]
        #pad_id_word=word_to_id["<PAD>"]
        #windows=self.config["windows"]
        feed_dict = self.create_feed_dict(is_train, batch)
        #print(feed_dict)
        #print(feed_dict[self.char_inputs])
        if is_train:
            global_step, loss, char_inputs, word_inputs, seg_inputs, pos_inputs, embedding, _ = sess.run(
                [self.global_step, self.loss, self.char_inputs, self.word_inputs, self.seg_inputs, self.pos_inputs, self.embedding, self.train_op],
                feed_dict)
            #print('XXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(char_inputs[0])
            #print(word_inputs[0])
            #print(seg_inputs[0])                      
            #emb = []
            #emb.append(char_inputs[0])
            #emb.append(word_inputs[0])
            #emb.append(seg_inputs[0])
            #embout = tf.concat(emb, axis=-1)
            #print(self.embedding_layer(char_inputs, word_inputs, seg_inputs))
            #print("number of vecs: %i" %(len(embedding[0][0])))

            #print('XXXXXXXXXXXXXXXXXXXXXXXXX')            
            #print(id_to_char[0])
            #print(id_to_word[0])

            #print("char_inputs: %i" %(len(char_inputs[0])))
            #print(id_to_char[char_inputs[0][0]])
            #print(embedding_char[0][0])
            #print("word_inputs: %i" %(len(word_inputs[0])))
            #print(id_to_word[word_inputs[0][0]])
            #print(embedding_word[0][0])
            #print()
            #print("seg_inputs: %i" %(len(seg_inputs[0])))
            #print(seg_inputs[0][0])
            #print(embedding_seg[0][0])
            #print("len of sentences 0: %i" %(len(embedding[0])))
            #print("number of vecs: %i" %(len(embedding[0][0])))
            #for i in embedding[0][0]:
            	#print(i)
            #for char in embedding[0]:
            	#print("number of vecs: %i" %(len(char)))
            	#for i in char:
            		#print(i)

            #for sen in embedding:
            	#print("len of sentences: %i" %(len(sen)))
            	#for char in sen:
            		#print("number of vecs: %i" %(len(char)))
            		#for i in char:
            			#print(i)
            
            #print(self.embedding_layer(char_inputs, word_inputs, seg_inputs))
            #print (embedding)
            
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            #print(batch)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
