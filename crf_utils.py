from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn

if str(tf.__version__).startswith("1.1.") or str(tf.__version__).startswith("1.2.") or str(tf.__version__).startswith(
        "1.3."):
    ##for tf1.1/1.2/1.3
    from tensorflow.contrib.rnn.python.ops import rnn_cell
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
else:
    ##for tf 1.4 and above
    from tensorflow.python.ops.rnn_cell import RNNCell as RNNCell
##

from tensorflow.python.ops import variable_scope as vs


# from tensorflow.contrib.crf.python.ops.crf import *;

import math

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
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

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
    state = array_ops.expand_dims(state, 2)

    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension. This performs the
    # multiplication of previous alpha values and the current binary potentials
    # in log space.
    transition_scores = state + self._transition_params
    new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])

    # Both the state and the output of this RNN cell contain the alphas values.
    # The output value is currently unused and simply satisfies the RNN API.
    # This could be useful in the future if we need to compute marginal
    # probabilities, which would require the accumulated alpha values at every
    # time step.
    return new_alphas, new_alphas

class CrfBackwardRnnCell(RNNCell):
  """Computes the beta values in a linear-chain CRF.
  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """

  def __init__(self, transition_params):
    """Initialize the CrfForwardRnnCell.
    Args:
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
    """
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfBackwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.
    Returns:
      new_betas, new_betas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """


    # This addition op broadcasts state along the second dimension, inputs along the second dimension.

    state = array_ops.expand_dims(state, 2)
    inputs = array_ops.expand_dims(inputs, 2)

    # different from forward, inputs should be added here

    all_scores = state + inputs + self._transition_params
    new_betas = math_ops.reduce_logsumexp(all_scores, [1])

    # Both the state and the output of this RNN cell contain the beta values.
    # The output value is currently unused and simply satisfies the RNN API.
    # This could be useful in the future if we need to compute marginal
    # probabilities, which would require the accumulated alpha values at every
    # time step.
    return new_betas, new_betas

def slot_probability(alphas,betas,logits,transtion_params,log_form,begin_index,end_index,list_of_tags,slot_boundary_check=None):

    """
    Specify some indices with some tags, we calculate the probability.
    Example: P(t_5 = tag_0, t_6 = tag_1) ==> which means the probability when 5th tag is tag_0 and 6th tag is tag_1.

    Args:
        alphas: here it's for single input, [max_seq_len, num_tags]
        betas:  here it's for single input, [max_seq_len, num_tags]
        logits: here it's for single input, [max_seq_len, num_tags]
        transtion_params: [num_tags, num_tags]
        log_form: log_form for probability calculation - Z
        begin_index: int, the start_index of slot
        end_index: int, the end index of slot
        list_of_tags: int, list of tags corresponding to each word in slot
        slot_boundary_check : int, the tag type that can't be the next of given slot.
    Returns:
        probability of slot
    """

    #pre-checks
    assert len(alphas) != 0 and len(betas) !=0 and len(logits) != 0, "alphas/betas/logits is in zero length"
    assert len(set([len(alphas),len(betas),len(logits)])) == 1 and len(set([len(alphas[0]),len(betas[0]),len(logits[0]),len(transtion_params[0]),len(transtion_params[1])])) == 1, "alphas/betas/logits/transition_params dimensions are not consistent"
    assert begin_index >=0 and begin_index < len(alphas), "begin index is out of boundary, %d"%(begin_index)
    assert end_index >=0 and end_index < len(alphas) and end_index>=begin_index, "end_index %d is illegal or smaller than begin index %d"%(end_index,begin_index)
    assert end_index-begin_index+1 == len(list_of_tags), "number of tags doesn't equal to end_index-begin_index+1"
    assert max(list_of_tags) < len(alphas[0]) and min(list_of_tags) >=0, "tag is out of boundary"
    assert log_form !=0, "log_form is 0"


    _log_score = 0.0
    previous_tag = -1
    max_seq_len = len(alphas)
    num_tags = len(alphas[0])

    # loop to calculate probability
    for tag_index,tag in enumerate(list_of_tags):
        seq_index = begin_index + tag_index

        # if it's first tag, use alpha
        if tag_index == 0:
            _log_score = alphas[seq_index][tag]
            if len(list_of_tags) == 1:
                _log_score += betas[seq_index][tag]
        else:
            _log_score += logits[seq_index][tag] + transtion_params[previous_tag][tag]

            # it's the last tag ,use beta
            if tag_index == len(list_of_tags)-1:
                if slot_boundary_check is None:
                    _log_score += betas[seq_index][tag]
                else:
                    if seq_index+1 >= max_seq_len: # means it's the end of query
                        _log_score += betas[seq_index][tag]
                    else:
                        exp_sum = 0.0

                        for i in range(num_tags):
                            if i != slot_boundary_check:
                                exp_sum += math.exp(logits[seq_index+1][i] + transtion_params[tag][i] + betas[seq_index+1][i])

                        _log_score += math.log(exp_sum,math.e)

        previous_tag = tag

    assert _log_score - log_form <= 0.00001, "the probability will be greater than 1, _log_score is %f, log_form is %f"%(_log_score,log_form)
    return math.exp(_log_score - log_form) if _log_score < log_form else 1.0

def crf_forward_backward_algo(inputs, sequence_lengths, transition_params):
    """Computes the alpha and beta for a CRF.
      Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
            to use as input to the CRF layer.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.
      Returns:
        alphas_seq: A [batch_size, max_seq_len, num_tags ]
        betas_seq: A [batch_size, max_seq_len, num_tags ]
        log_norm: A [batch_size] vector of normalizers for a CRF.

      """
    def _single_seq_fn():
        batch_size = array_ops.shape(inputs)[0]
        num_tags = array_ops.shape(inputs)[2]

        alphas_seq = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])

        betas = tf.constant(0.0, shape=[1, 1])  # as we use log, so 0.0 for beta initialization
        betas = tf.tile(betas, multiples=[batch_size, num_tags])
        betas_seq = array_ops.expand_dims(betas, 1)
        # crf log norm
        log_norm = math_ops.reduce_logsumexp(alphas_seq, [2])

        return alphas_seq,betas_seq,log_norm

    def _multi_seq_fn():
        # Split up the first and rest of the inputs in preparation for the forward
        # algorithm.
        batch_size = array_ops.shape(inputs)[0]
        num_tags = array_ops.shape(inputs)[2]

        first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
        first_input = array_ops.squeeze(first_input, [1])
        rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

        # Compute the alpha values in the forward algorithm
        forward_cell = CrfForwardRnnCell(transition_params)
        alphas_seq, alphas = rnn.dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths - 1,
            initial_state=first_input,
            dtype=dtypes.float32)
        # Get all alphas in each time steps
        alphas_seq = tf.concat([tf.expand_dims(first_input,axis=1),alphas_seq],axis=1)

        # Compute the betas values in the backward algorithm
        first_input = tf.constant(0.0,shape=[1,1]) # as we use log, so 0.0 for beta initialization
        first_input = tf.tile(first_input,multiples=[batch_size,num_tags])

        # reverse the sequence of inputs in forward algorithm for backward algorithm
        rest_of_input = gen_array_ops.reverse_sequence(rest_of_input,sequence_lengths-1,seq_dim=1)

        # transpose transition parameters for backward algorithm
        backward_cell = CrfBackwardRnnCell(tf.transpose(transition_params,perm=[1,0]))
        betas_seq, betas = rnn.dynamic_rnn(cell=backward_cell,inputs = rest_of_input,sequence_length=sequence_lengths - 1,
            initial_state=first_input,
            dtype=dtypes.float32)

        betas_seq = tf.concat([tf.expand_dims(first_input, axis=1), betas_seq], axis=1)

        # reverse betas that follows same index as alphas
        betas_seq = tf.reverse_sequence(betas_seq,sequence_lengths,seq_dim=1)

        # crf log norm
        log_norm = math_ops.reduce_logsumexp(alphas, [1])

        return alphas_seq,betas_seq,log_norm

    return utils.smart_cond(
        pred=math_ops.equal(
            inputs.shape[1].value or array_ops.shape(inputs)[1], 1),
        fn1=_single_seq_fn,
        fn2=_multi_seq_fn)


class CrfNbestDecodeForwardRnnCell(RNNCell):
    """Computes the forward decoding in a linear-chain CRF.
    """

    def __init__(self, transition_params, K):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, 1, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._transition_params = array_ops.expand_dims(self._transition_params, 2)
        self._num_tags = transition_params.get_shape()[0].value
        self._K = K

    @property
    def state_size(self):
        return self._num_tags * self._K

    @property
    def output_size(self):
        return self._num_tags * self._K

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags * K] matrix containing the previous step's
                score values.
          scope: Unused variable scope of this cell.
        Returns:
          backpointers: A [batch_size, num_tags * K] matrix of backpointers.
          new_state: A [batch_size, num_tags * K] matrix of new score values.
        """
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output), 'number of N-best' by 'K'.
        # shape of state becomes [B, O * K]

        batch_size = array_ops.shape(state)[0]

        # This expand the last dimension of state for later broadcast usage with transition parameters
        state = tf.reshape(state, [batch_size, self._num_tags, -1])
        state = array_ops.expand_dims(state, 3)  # [B, O, K, 1]

        # This addition op broadcasts self._transitions_params along the zeroth & third
        # dimension and state along the last dimension.
        # [B, O, K, 1] + [1, O, 1, O] -> [B, O, K, O]
        transition_scores = state + self._transition_params  # [B, O, K, O]

        # here reshape transition_scores for top_k and backpointers usage
        transition_scores = tf.reshape(transition_scores, [batch_size, -1, self._num_tags])  # [B, O*K, O]
        transition_scores = tf.transpose(transition_scores, perm=[0, 2, 1])  # [B, O, O*K]

        # get Top_k
        values, indices = tf.nn.top_k(transition_scores, k=array_ops.shape(state)[2])

        # new_state = array_ops.expand_dims(inputs,-1) + math_ops.reduce_max(transition_scores, [1])  # [B, O, K]
        new_state = array_ops.expand_dims(inputs, axis=-1) + values  # [B, O, K]
        backpointers = math_ops.cast(indices, dtype=dtypes.int32)  # [B, O, K]

        new_state = tf.reshape(new_state, [batch_size, -1])
        backpointers = tf.reshape(backpointers, [batch_size, -1])

        return backpointers, new_state


class CrfNbestDecodeBackwardRnnCell(RNNCell):
    """Computes backward decoding in a linear-chain CRF.
    """

    def __init__(self, num_tags, K):
        """Initialize the CrfDecodeBackwardRnnCell.
        Args:
          num_tags: An integer. The number of tags.
          K: number of hypothesis to output
        """
        self._num_tags = num_tags
        self._K = K

    @property
    def state_size(self):
        return self._K

    @property
    def output_size(self):
        return self._K

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
        Args:
          inputs: A [batch_size, num_tag * K] matrix of
                backpointer of next step (in time order).
          state: A [batch_size, K] matrix of tag index of next step.
          scope: Unused variable scope of this cell.
        Returns:
          new_tags, new_tags: A pair of [batch_size, K]
            tensors containing the new tag indices.
        """
        # state = array_ops.squeeze(state, axis=[1])                # [B]

        batch_size = array_ops.shape(inputs)[0]
        b_indices = math_ops.range(batch_size)  # [B]
        b_indices = tf.tile(array_ops.expand_dims(b_indices, axis=0), [array_ops.shape(state)[1], 1])  # [K, B]
        b_indices = tf.transpose(b_indices, perm=[1, 0])  # [B, K]

        indices = array_ops.stack([tf.reshape(b_indices, [-1]), tf.reshape(state, [-1])], axis=1)  # [B * K, 2]
        new_tags = array_ops.reshape(
            gen_array_ops.gather_nd(inputs, indices),  # [B * K]
            [batch_size, -1])  # [B, K]

        return new_tags, new_tags


def crf_one_case_nbest_decode(potentials, transition_params, sequence_length, K):
    """Decode the top k scoring sequence of tags in TensorFlow.
    @@TODO: this is a function for batch_size = 1, if batch_size is not 1 and sequence length are variant, we might have different Ks for different cases due to length variance.
    This is a function for tensor.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
      K : the number of hypothesis to output
    Returns:
      decode_tags: A [batch_size, K, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the top K scoring tag indices.
      best_score: A [batch_size, K ] vector, containing the score of `decode_tags`.
    """

    # If max_seq_len is 1, we skip the algorithm and simply return the top k  tag
    # and the top k  activation.
    def _single_seq_fn():
        squeezed_potentials = array_ops.squeeze(potentials, [1])
        batch_size = array_ops.shape(potentials)[0]
        num_tags = potentials.get_shape()[2].value

        # if K > num_tag ^ seq_len ( all possible hypothesis), use num_tag ^ seq_len as to trim
        log_cnt_total_cases = tf.cast(sequence_length[0], dtypes.float32) * tf.log(
            tf.cast(tf.constant(num_tags), dtypes.float32))
        K_modified = tf.cond(
            tf.logical_or(tf.less(tf.log(tf.cast(tf.constant(K), dtypes.float32)), log_cnt_total_cases),
                          tf.less(log_cnt_total_cases, tf.constant(0.0))), lambda: tf.constant(K),
            lambda: tf.pow(tf.constant(num_tags), sequence_length[0]))

        # K_modified = tf.constant(K)

        top_K_values, top_K_indices = tf.nn.top_k(squeezed_potentials, K_modified)

        decode_tags = array_ops.expand_dims(
            top_K_indices, 2)
        decode_tags = math_ops.cast(decode_tags, dtype=dtypes.int32)  # [B, K, 1]

        best_score = top_K_values  # [B, K]
        return decode_tags, best_score

    def _multi_seq_fn():
        """Decoding of highest scoring sequence."""

        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        num_tags = potentials.get_shape()[2].value
        batch_size = array_ops.shape(potentials)[0]
        # Computes forward decoding. Get last score and backpointers.
        crf_fwd_cell = CrfNbestDecodeForwardRnnCell(transition_params, K)
        initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
        # initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]

        # Padding initital state to fit N-best format
        modified_initial_state = tf.transpose(initial_state, perm=[0, 2, 1])
        padding_for_init_state = tf.constant(-1.0e38, shape=[1, 1, 1])
        padding_for_init_state = tf.tile(padding_for_init_state, multiples=[batch_size, num_tags, K - 1])

        modified_initial_state = tf.concat([modified_initial_state, padding_for_init_state], axis=2)

        modified_initial_state = tf.reshape(modified_initial_state,
                                            shape=[array_ops.shape(potentials)[0], -1])  # [B, O*K]

        inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]

        # follow dynamic_rnn logic as a dynamic programming to get TopKs in each step
        backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O*K], [B, O*K]
            crf_fwd_cell,
            inputs=inputs,
            sequence_length=sequence_length - 1,
            initial_state=modified_initial_state,
            time_major=False,
            dtype=dtypes.int32)

        backpointers = gen_array_ops.reverse_sequence(  # [B, T - 1, O*K]
            backpointers, sequence_length - 1, seq_dim=1)

        # Computes backward decoding. Extract tag indices from backpointers.
        crf_bwd_cell = CrfNbestDecodeBackwardRnnCell(num_tags, K)

        top_K_values, top_K_indices = tf.nn.top_k(last_score, K)

        initial_state = math_ops.cast(top_K_indices,  # [B, K]
                                      dtype=dtypes.int32)

        decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, K]
            crf_bwd_cell,
            inputs=backpointers,
            sequence_length=sequence_length - 1,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)

        initial_state = array_ops.expand_dims(initial_state, axis=[1])  # [B, 1, K]
        decode_tags = array_ops.concat([initial_state, decode_tags],  # [B, T, K]
                                       axis=1)
        decode_tags = gen_array_ops.reverse_sequence(  # [B, T, K]
            decode_tags, sequence_length, seq_dim=1)

        # if K > num_tag ^ seq_len ( all possible hypothesis), use num_tag ^ seq_len as to trim
        log_cnt_total_cases = tf.cast(sequence_length[0], dtypes.float32) * tf.log(
            tf.cast(tf.constant(num_tags), dtypes.float32))
        K_modified = tf.cond(
            tf.logical_or(tf.less(tf.log(tf.cast(tf.constant(K), dtypes.float32)), log_cnt_total_cases),
                          tf.less(log_cnt_total_cases, tf.constant(0.0))), lambda: tf.constant(K),
            lambda: tf.pow(tf.constant(num_tags), sequence_length[0]))
        # K_modified = tf.constant(K)

        decode_tags = tf.transpose(decode_tags, perm=[0, 2, 1])  # [B, K, T]
        decode_tags = decode_tags / tf.constant(K)
        decode_tags = tf.floor(decode_tags)
        decode_tags = math_ops.cast(decode_tags, dtype=dtypes.int32)

        best_score = top_K_values  # [B, K]

        decode_tags = tf.slice(decode_tags, [0, 0, 0], [-1, K_modified, -1])
        best_score = tf.slice(best_score, [0, 0], [-1, K_modified])

        return decode_tags, best_score

    # return _multi_seq_fn()



    return utils.smart_cond(
        pred=math_ops.equal(
            potentials.shape[1].value or array_ops.shape(potentials)[1], 1),
        fn1=_single_seq_fn,
        fn2=_multi_seq_fn)
