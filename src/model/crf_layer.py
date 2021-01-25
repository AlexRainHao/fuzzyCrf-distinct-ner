import tensorflow as tf
from tensorflow.nn.rnn_cell import RNNCell
from tensorflow.nn import dynamic_rnn

class CrfForwardRnnCell(RNNCell):
    """Computes the alpha values in a linear-chain CRF.

    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """
    
    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.

        Args:
          transition_params: A [num_tags, num_tags] matrix of binary potentials.
              This matrix is expanded into a [1, num_tags, num_tags] in preparation
              for the broadcast summation occurring within the cell.
        """
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = tf.dimension_value(transition_params.shape[0])
    
    @property
    def state_size(self):
        return self._num_tags
    
    @property
    def output_size(self):
        return self._num_tags
    
    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.

        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
              values.
          scope: Unused variable scope of this cell.

        Returns:
          new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
              values containing the new alpha values.
        """
        state = tf.expand_dims(state, 2)
        
        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension. This performs the
        # multiplication of previous alpha values and the current binary potentials
        # in log space.
        transition_scores = state + self._transition_params
        new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])
        
        # Both the state and the output of this RNN cell contain the alphas values.
        # The output value is currently unused and simply satisfies the RNN API.
        # This could be useful in the future if we need to compute marginal
        # probabilities, which would require the accumulated alpha values at every
        # time step.
        return new_alphas, new_alphas


class fuzzyCrfForwardCell(RNNCell):
    
    def __init__(self, transition):
        self._transition = tf.expand_dims(transition, 0)
        self._num_tags = tf.dimension_value(transition.shape[0])
    
    @property
    def state_size(self):
        return self._num_tags
    
    @property
    def output_size(self):
        return self._num_tags
    
    def __call__(self, inputs, state, scope=None):
        # is_single # [batch]
        # feature # [batch, num_tag]
        # tag # [batch, num_tag]
        
        feature, tag, is_single = inputs
        
        transition_score = tf.multiply(tf.expand_dims(state, 2) + self._transition,
                                       is_single)
        transition_score = feature + tf.reduce_logsumexp(transition_score, axis=[1])
        transition_score = tf.multiply(transition_score, tf.cast(tag, tf.float32))  # [batch, num_tag]
        
        return transition_score, transition_score



def crf_log_norm(inputs, sequence_lengths, transition_params):
    """Computes the normalization for a CRF.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])
    
    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
    # the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                            tf.zeros_like(log_norm),
                            log_norm)
        return log_norm
    
    def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        
        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.
        forward_cell = CrfForwardRnnCell(transition_params)
        # Sequence length is not allowed to be less than zero.
        sequence_lengths_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_lengths.dtype),
            sequence_lengths - 1)
        
        _, alphas = dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths_less_one,
            initial_state=first_input,
            dtype=tf.float32)
        log_norm = tf.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                            tf.zeros_like(log_norm),
                            log_norm)
        return log_norm
    
    return tf.cond(
        pred=tf.equal(
            tf.dimension_value(
                inputs.shape[1]) or tf.shape(inputs)[1],
            1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)