"""
pass

data sequence example:
    A B C D E F G H I J K L
    O O B-LOC I-LOC I-LOC O O UNK UNK O O O
    O O B-EDU I-EDU I-EDU O O UNK UNK O O O

for a single sequence X[x1, x2, ..., xn]
corresponds to Y [0, 0, [1,4], [2,5], [3,6], 0, 0, -1, -1, 0, 0, 0]
"""
from typing import Dict, List, Text, Any, Union
import tensorflow as tf
import numpy as np

ConstUnkStr = "UNK"

def sequence_mask_1D(seq_len, max_len):
    """only accept 1D list"""
    seq_len = np.reshape(np.asarray(seq_len), -1)
    maxq = max(seq_len)
    
    assert maxq <= max_len, "max length assigned should greater than each of sequence length"
    
    mat = np.zeros(shape = (seq_len.shape[0], max_len))
    for idx, seq in enumerate(seq_len):
        mat[idx, :seq] = 1
    
    return mat

class fuzzyCrf:
    """pass"""
    
    def __init__(self, tag2int: Dict):
        self.tag2int = tag2int
        pass
    
    @property
    def UNK(self):
        """pass"""
        return -1
    
    @property
    def Tags(self):
        """pass"""
        return list(self.tag2int.values())
    
    def _convert_unk(self, tag_index):
        """pass"""
        if tag_index == self.UNK:
            return self.Tags

        else:
            return np.asarray(tag_index).reshape(-1)
    
    def _forward_unary_score(self, tag_indices, fea_vec, mask):
        """pass"""
        last_score = np.zeros(1) # start node score
        for node, (tag, fea) in enumerate(zip(tag_indices, fea_vec)):
            if not mask[node]:
                break
                
            last_axis = 1 if tag == self.UNK else 0
            tag = self._convert_unk(tag)
            last_score = np.sum(np.expand_dims(last_score, last_axis) + fea_vec[node, tag], axis = 0)
            
            if isinstance(tag, int):
                last_score = last_score.sum(keepdims = True)
            
        return last_score
    
    def _forwar_binary_score(self, tag_indices, trans_mat, mask):
        """pass"""
        last_score = np.zeros(1)
        tag_pairs = list(zip(*[tag_indices[i:] for i in range(2)]))
        
        for node, (s_tag, e_tag) in enumerate(tag_pairs):
            if not mask[node + 1]:
                break
            
            # last_axis = 1 if (s_tag == self.UNK or e_tag == self.UNK) else 0
            last_axis = 1 if (isinstance(s_tag, list) and isinstance(e_tag, list)) else 0
            s_tag = self._convert_unk(s_tag)
            e_tag = self._convert_unk(e_tag)
        
            if last_axis:
                last_score += trans_mat[s_tag, e_tag]
            else:
                last_score = np.sum(trans_mat[np.ix_(s_tag, e_tag)] + np.expand_dims(last_score, 1), axis = 0)

        return last_score
    
    def unary_score(self, tag_indices, sequence_length, fea_vec):
        """
        a dp-alg calculation for series of paths emission score
        Parameters
        ----------
        tag_indices: A [batch_size, max_seq_len] matrix
        sequence_length: A [batch_size] vector
        fea_vec: A [batch_size, max_seq_len, num_tags] tensor
        """
        max_seq_len = fea_vec.shape[1]
        
        masks = sequence_mask_1D(sequence_length, max_seq_len)

        unary_score = np.fromiter(map(lambda tag, fea, mask: self._forward_unary_score(tag, fea, mask),
                                      tag_indices, fea_vec, masks),
                                  dtype = np.float32)  # [batch]

        return unary_score

    def binary_score(self, tag_indices, sequence_length, transition_mat):
        """
        a dp-alg calculation for series of paths transition score
        Parameters
        ----------
        tag_indices: A [batch_size, max_seq_len] matrix
        sequence_length: A [batch_size] vector
        transition_mat: A [num_tags, num_tags] transition probability matrix
        """
        max_seq_len = tag_indices.shape[1]
        
        masks = sequence_mask_1D(sequence_length, max_seq_len)
        
        binary_score = np.fromiter(map(lambda tag, mask: self._forwar_binary_score(tag, transition_mat, mask),
                                       tag_indices, masks),
                                   dtype = np.float32) # [batch]
        
        return binary_score
    
    def sequence_score(self, *args, **kwargs):
        """pass"""
        
        raise NotImplementedError()

import random

fea_place = tf.placeholder(tf.float32, shape = [None, None], name = "fea_place")
tag_place = tf.placeholder(tf.string, name = "tag_place")
len_place = tf.cast(tf.reduce_sum(tf.sign(tf.abs(fea_place)), reduction_indices = 1), tf.int32)

loss = tf.constant(0.0, name = "loss")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3)
# trian_op = optimizer.minimize(loss)


tag2int = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6}

len_data = [10, 6, 7, 12, 2]
tag_data = "[0, 0, [1,4], [2,5], [3,6], 0, 0, -1, -1, 0, 0, 0]"

transition_mat = np.random.random(size = (7, 7))
op = fuzzyCrf(tag2int)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for i in range(10):
fea_vec = np.random.random(size = (12, 7))
loss = sess.run(loss, feed_dict = {fea_place: fea_vec,
                                   tag_place: tag_data})



# if __name__ == '__main__':
    # tag2int = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6}
    # op = fuzzyCrf(tag2int)
    # seq_len = np.asarray([10, 6, 7, 12, 2])
    # tag_index = [[0, 0, [1,4], [2,5], [3,6], 0, 0, -1, -1, 0, 0, 0]] * 5
    # fea_vec = np.random.random(size = (5, 12, 7))
    # res = op._unary_score(tag_index, seq_len, fea_vec)
    # print(res)

