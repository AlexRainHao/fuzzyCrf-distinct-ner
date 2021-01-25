# TODO: length == 1 checkout for each sub-function

from typing import Dict, List, Text, Any, Union
import tensorflow as tf
from tensorflow.nn import dynamic_rnn
import numpy as np

from src.model.crf_layer import fuzzyCrfForwardCell, \
    CrfDecodeForwardRnnCell, \
    CrfDecodeBackwardRnnCell, \
    crf_log_norm

class fuzzyCrf:

    defaults = {
        "learning_rate": 1e-3
    }

    def __init__(self, tag2int: Dict, learning_rate = None):
        self.tag2int = tag2int
        self.learning_rate = learning_rate or self.defaults.get("learning_rate")

    @property
    def tag_size(self):
        return len(self.tag2int)
    
    @property
    def UNK(self):
        return tf.constant(-1, dtype=tf.int32)
    
    def build_graph(self):
        """pass"""
        g = tf.Graph()
        with g.as_default():
            self._create_placeholder()
            self._create_transition()
            self._forward_score()
            self._create_loss()
            self._create_train_layer()
        return g
    
    def _create_placeholder(self):
        """pass"""
        self.tag_place = tf.placeholder(tf.int32, [None, None, self.tag_size], name="tag_place")
        self.token_vec_place = tf.placeholder(tf.float32, [None, None, self.tag_size], name="token_vec_place")
        self.seq_len_place = tf.placeholder(tf.int32, [None], name="seq_len_place")
        self.is_unk = tf.placeholder(tf.int32, [None, None], name = "is_unk")
        self.pred_place = tf.placeholder(tf.float32, [None], name="pred_place")

    def _create_transition(self):
        """pass"""
        self.transitions = tf.get_variable("transitions",
                                           shape=[self.tag_size, self.tag_size],
                                           initializer=tf.contrib.layers.xavier_initializer())

    def _forward_score(self):
        """pass"""
        with tf.variable_scope("fuzzy_crf_forward"):
            first_fea = tf.squeeze(tf.slice(self.token_vec_place, [0, 0, 0], [-1, 1, -1]), [1])
            first_tag = tf.squeeze(tf.slice(self.tag_place, [0, 0, 0], [-1, 1, -1]), [1]) # [batch, num_tag]

            first_state = tf.multiply(first_fea, tf.cast(first_tag, tf.float32)) # [batch, num_tag]

            # ===========
            rest_fea = tf.slice(self.token_vec_place, [0, 1, 0], [-1, -1, -1])
            rest_tag = tf.slice(self.tag_place, [0, 1, 0], [-1, -1, -1])
            rest_unk = self._sinpath_mask() # [batch, max_seq_len - 1, num_tag, num_tag]

            forward_cell = fuzzyCrfForwardCell(self.transitions)
            sequence_lengths_less_one = tf.maximum(
                tf.constant(0, dtype=self.seq_len_place.dtype),
                self.seq_len_place - 1)

            _, scores = dynamic_rnn(
                cell = forward_cell,
                inputs = (rest_fea, rest_tag, rest_unk),
                sequence_length = sequence_lengths_less_one,
                initial_state = first_state,
                dtype = tf.float32
            )

            self.forward_score = tf.reduce_logsumexp(scores, [1]) # [batch]

    def _create_loss(self):
        """pass"""
        with tf.variable_scope("loss_layer"):
            norm_score = crf_log_norm(self.token_vec_place, self.seq_len_place, self.transitions) # [batch]
            log_likelihood = self.forward_score - norm_score #[batch]

            self.loss = tf.reduce_mean(-log_likelihood)

    def _create_train_layer(self):
        """pass"""
        with tf.variable_scope("train_layer"):
            self.train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def _is_sinpath_mask(self, unk):
        """pass"""
        unk = tf.cast(unk, tf.bool)

        _ones_mask = tf.ones((self.tag_size, self.tag_size), dtype = tf.float32)
        _diag_mask = tf.cast(tf.diag(np.repeat(1., self.tag_size)), dtype = tf.float32)

        return tf.cond(unk,
                       true_fn = lambda: _ones_mask,
                       false_fn = lambda: _diag_mask)

    def _sinpath_mask(self):
        """pass"""
        batch_size = tf.shape(self.tag_place)[0]
        max_seq_len = tf.shape(self.tag_place)[1]

        # if max_seq_len == 1:
        #     pass

        head_unk = tf.slice(self.is_unk, [0, 0], [-1, max_seq_len - 1]) # [batch, len - 1]
        tail_unk = tf.slice(self.is_unk, [0, 1], [-1, -1])
        con_unk = tf.reshape(tf.add(head_unk, tail_unk), [-1])

        masks = tf.map_fn(self._is_sinpath_mask, con_unk, dtype = tf.float32)
        masks = tf.reshape(masks, [batch_size, max_seq_len - 1, self.tag_size, self.tag_size])

        return masks

    def decode(self, fea_vec, transitions, seq_len):
        """pass"""
        forward_cell = CrfDecodeForwardRnnCell(transitions)

        first_vec = tf.squeeze(tf.slice(fea_vec, [0, 0, 0], [-1, 1, -1]), [1]) # [batch, num_tag]
        rest_vec = tf.squeeze(tf.slice(fea_vec, [0, 1, 0], [-1, -1, -1])) # [batch, len - 1, num_tag]

        sequence_lengths_less_one = tf.maximum(
            tf.constant(0, dtype=seq_len.dtype),
            seq_len - 1)

        backpointers, scores = dynamic_rnn(
            cell = forward_cell,
            inputs = rest_vec,
            sequence_length = sequence_lengths_less_one,
            initial_state = first_vec,
            dtype = tf.int32
        )
        backpointers = tf.reverse_sequence(backpointers, sequence_lengths_less_one, seq_dim = 1) # [batch, len - 1, num_tag]

        # ===================
        num_tags = tf.dimension_value(tf.shape(transitions)[1])
        backward_cell = CrfDecodeBackwardRnnCell(num_tags)

        init_state = tf.cast(tf.argmax(scores, axis = 1), dtype = tf.int32)
        init_state = tf.expand_dims(init_state, axis = 1) # [batch, 1]

        decode_tags, _ = dynamic_rnn(
            cell = backward_cell,
            inputs = backpointers,
            sequence_length = sequence_lengths_less_one,
            dtype = tf.int32
        )
        decode_tags = tf.squeeze(decode_tags, axis = [-1]) # [batch, len - 1]
        decode_tags = tf.concat([init_state, decode_tags], axis = 1) # [batch, len]
        decode_tags = tf.reverse_sequence(decode_tags, seq_len, seq_dim = 1)

        best_score = tf.reduce_max(scores, axis = [1])

        return decode_tags, best_score


if __name__ == '__main__':
    tag2int = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6}
    seq_len = np.asarray([12] * 5)
    tag_index = np.random.randint(0, 2, size=(5, 12, 7))
    fea_vec = np.random.random(size=(5, 12, 7))
    pred_vec = np.random.random(size=(5))
    is_unk = np.all(tag_index, axis = -1)


    op = fuzzyCrf(tag2int=tag2int)

    g = op.build_graph()

    sess = tf.InteractiveSession(graph=g)
    sess.run(tf.global_variables_initializer())

    for i in range(5):
        _, loss = sess.run([op.train_op, op.loss], feed_dict={
            op.tag_place: tag_index,
            op.token_vec_place: fea_vec,
            op.seq_len_place: seq_len,
            op.is_unk: is_unk,
            op.pred_place: pred_vec})

    tags, _ = op.decode(fea_vec = fea_vec,
                        transitions = op.transitions,
                        seq_len = seq_len)
    print(tags.eval())

