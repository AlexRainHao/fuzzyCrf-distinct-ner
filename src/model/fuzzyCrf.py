# TODO: length == 1 checkout for each sub-function

from typing import Dict, List, Text, Any, Union
import tensorflow as tf
from tensorflow.nn import dynamic_rnn
import numpy as np

from src.model.crf_layer import fuzzyCrfForwardCell, crf_log_norm

class fuzzyCrf:
    
    def __init__(self, tag2int: Dict):
        self.tag2int = tag2int
        
    @property
    def tag_size(self):
        return len(self.tag2int)
    
    @property
    def UNK(self):
        return tf.constant(-1, dtype=tf.int32)
    
    def build_graph(self):
        g = tf.Graph()
        with g.as_default():
            self._create_placeholder()
            self._create_transition()
            self._forward_score()
        return g
    
    def _create_placeholder(self):
        self.tag_place = tf.placeholder(tf.int32, [None, None, self.tag_size], name="tag_place")
        self.token_vec_place = tf.placeholder(tf.float32, [None, None, self.tag_size], name="token_vec_place")
        self.seq_len_place = tf.placeholder(tf.int32, [None], name="seq_len_place")
        self.is_unk = tf.placeholder(tf.int32, [None, None], name = "is_unk")
        self.pred_place = tf.placeholder(tf.float32, [None], name="pred_place")

    def _create_transition(self):
        self.transitions = tf.get_variable("transitions",
                                           shape=[self.tag_size, self.tag_size],
                                           initializer=tf.contrib.layers.xavier_initializer())

    def _forward_score(self):
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
        
        _, alphas = dynamic_rnn(
            cell = forward_cell,
            inputs = (rest_fea, rest_tag, rest_unk),
            sequence_length = sequence_lengths_less_one,
            initial_state = first_state,
            dtype = tf.float32
        )
        
        forward_scores = tf.reduce_logsumexp(alphas, [1])
        norm_score = self._norm_score()

        self.score = tf.identity(forward_scores - norm_score, name="score")

    def _norm_score(self):
        return crf_log_norm(self.token_vec_place, self.seq_len_place, self.transitions) # [batch]

    def _is_sinpath_mask(self, unk):
        unk = tf.cast(unk, tf.bool)
        
        return tf.cond(unk,
                       true_fn = lambda: tf.ones((self.tag_size, self.tag_size), dtype = tf.float32),
                       false_fn = lambda: tf.diag(tf.repeat(1., self.tag_size)))
    
    def _sinpath_mask(self):
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

    scores = sess.run([op.score], feed_dict={
        op.tag_place: tag_index,
        op.token_vec_place: fea_vec,
        op.seq_len_place: seq_len,
        op.is_unk: is_unk,
        op.pred_place: pred_vec})

    print(scores)
