# -*- coding: utf-8 -*-
# =============================================================================
# LSTM 2核心代码 拓扑结构1
# 作者：张璐
# 时间：2018-03-25
# =============================================================================

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale


class InputProducer(object):
    def __init__(self, series, config):
        self.series = scale(series)
        self.length = len(series)

    # seq2point producer
    # shift: 输出偏移幅度
    def next_batch(self, batch_size, num_steps, input_size, output_size, shift):
        batch_len = batch_size * (input_size * num_steps + shift)

        i = tf.train.range_input_producer(self.length - batch_len, shuffle=False).dequeue()
        data = tf.strided_slice(self.series, [i], [i + batch_len])
        data = tf.reshape(data, [batch_size, -1])
        
        x = tf.slice(data, [0, 0], [batch_size, num_steps * input_size])
        y = tf.slice(data, [0, shift], [batch_size, num_steps * output_size])
        return x, y



class LSTModel(object):
    def __init__(self, is_training, config, input_):
        batch_size = config.batch_size
        num_steps = config.num_steps
        input_size = config.input_size
        hidden_size = config.hidden_size
        output_size = config.output_size

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float64)

        weights = {
            "hidden": tf.get_variable("hidden_w", [input_size, hidden_size],
                                      initializer=tf.random_normal_initializer(), dtype=tf.float64),
            "output": tf.get_variable("output_w", [hidden_size, output_size],
                                      initializer=tf.random_normal_initializer(), dtype=tf.float64)
        }

        biases = {
            "hidden": tf.get_variable("hidden_b", [hidden_size],
                                      initializer=tf.constant_initializer(0.0), dtype=tf.float64),
            "output": tf.get_variable("output_b", [output_size],
                                      initializer=tf.constant_initializer(0.0), dtype=tf.float64)
        }
        
        self._x, self._y = input_.next_batch(batch_size, num_steps, input_size, output_size, config.shift)

        if is_training and config.keep_prob < 1:
            self._x = tf.nn.dropout(self._x, config.keep_prob)

        _X = tf.reshape(self._x, [-1, input_size])
        _X = tf.matmul(_X, weights["hidden"]) + biases["hidden"]

        if is_training and config.keep_prob < 1:
            _X = tf.nn.dropout(_X, keep_prob=config.keep_prob)

        _X = tf.reshape(_X, [batch_size, num_steps, hidden_size])
        unstack_x = tf.unstack(_X, num=num_steps, axis=1)

        outputs, states = tf.contrib.rnn.static_rnn(cell, unstack_x, initial_state=self._initial_state)

        # outputs = outputs[-1:]

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
        self._predict = tf.matmul(output, weights["output"]) + biases["output"]

        if is_training and config.keep_prob < 1:
            self._predict = tf.nn.dropout(self._predict, keep_prob=config.keep_prob)

        if is_training is False:
            return
        
        loss = tf.reduce_mean(tf.square(self._y - self._predict))
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._cost = tf.reduce_sum(loss) / batch_size


    @property
    def cost(self):
        return self._cost

    @property
    def predict(self):
        return self._predict

    @property
    def input(self):
        return self._x

    @property
    def valid(self):
        return self._y

    @property
    def optimizer(self):
        return self._optimizer

