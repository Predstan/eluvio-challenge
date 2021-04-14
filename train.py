"""
Class Implementation of Model for Predicting Eluvio Dataset.
Please note That unit testing were not performed and 
may throw the slightest error if not properly used
Refer to the note.ipynb Notebook for proper Usage

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import random
import re
import sys
import os

import numpy as np
from six.moves import xrange  
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import tensorflow.compat.v1.keras.backend as K
tf.compat.v1.disable_eager_execution()


class Model:
    def __init__(self,input_size, first_dense, model_dir, third_dense, second_dense, output_size, sess = False):
        self._input_size = input_size
        self._output_size = output_size
        self._model_dir = model_dir
        self._first_dense = first_dense
        self._second_dense = second_dense
        self._third_dense = third_dense
        self._loaded = False
        self._start_step = 0
        self._global_step = tf.compat.v1.train.get_or_create_global_step()
        self._save_step = 1



        if sess is False:
            self._sess = tf.compat.v1.InteractiveSession()
        else:
            self._sess = sess

        
        self._final_layer, self._dropout = self._build(input_size, first_dense, second_dense, third_dense, output_size, training=True)
        
        self.train(learn_rate=[0,0], dropout_rate=0, save_step=0, batch_size=0, eval_step=0,
                    training_time=0, rate_step=0, display_step=0, train_data=0, Validation_data=0, init=True)
       


    
    def _build(self, input_size, first_dense, second_dense, third_dense, output_size, training=True, input_1d=None):
        """
            Builds the Model Graph
            Args:
                input_size: Size of the Input Vector
                first_Dense: Number of Neurons in First Dense Layer
                second_dense: Number of Neurons in Second Dense Layer
                third_dense: Number of Neurons in third Dense Layer
                output_size: Number of Categorical Labels
                training: If training is True, add dropout layers, else: do not add the layer
                input_1d: Saving a pb model will not work with model build with placeholders 
                            hence input is a reshape input see saved_pb_model method
            Return:
                Last Layer for Softmax and dropout if training else: Last Layer only
            """


        dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
        
        
        self._input = tf.compat.v1.placeholder(
            tf.float32, [None, input_size], name='data_input')

        if training:
            input_reshape = tf.reshape(self._input,                      # input: [batch_size, input_size]
                                    [-1, input_size]) 
        else:
            input_reshape = tf.reshape(input_1d,                      # input: [batch_size, input_size]
                                    [-1, input_size])

        with tf.compat.v1.variable_scope("first_weights", reuse=tf.compat.v1.AUTO_REUSE):

            first_weights = tf.compat.v1.get_variable(                          # Weights Initialization 
                name='first_weights',
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                shape=[input_size, first_dense])

        with tf.compat.v1.variable_scope("first_bias", reuse=tf.compat.v1.AUTO_REUSE):
            first_bias = tf.compat.v1.get_variable(                              # Bias Initialization 
                name='first_bias',
                initializer=tf.compat.v1.zeros_initializer,
                shape=[first_dense,])

        first_dense_layer = tf.matmul(input_reshape, first_weights) + first_bias

        first_relu = tf.nn.relu(first_dense_layer)

        if training:
            first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)
        else:
            first_dropout = first_relu

        first_out_shape = first_dropout.shape[1]
        print("first shape", first_out_shape)
        with tf.compat.v1.variable_scope("second_weights", reuse=tf.compat.v1.AUTO_REUSE):

            second_weights = tf.compat.v1.get_variable(                          # Weights Initialization 
                name='second_weights',
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                shape=[first_out_shape, second_dense])

        with tf.compat.v1.variable_scope("second_bias", reuse=tf.compat.v1.AUTO_REUSE):
            second_bias = tf.compat.v1.get_variable(                              # Bias Initialization 
                name='second_bias',
                initializer=tf.compat.v1.zeros_initializer,
                shape=[second_dense,])

        second_dense_layer = tf.matmul(first_dropout, second_weights) + second_bias

        second_relu = tf.nn.relu(second_dense_layer)

        if training:
            second_dropout = tf.nn.dropout(second_relu, rate=dropout_rate)
        else:
            second_dropout = second_relu

        second_out_shape = second_dropout.shape[1]
        print("second shape", second_out_shape)

        with tf.compat.v1.variable_scope("third_weights", reuse=tf.compat.v1.AUTO_REUSE):

            third_weights = tf.compat.v1.get_variable(                          # Weights Initialization 
                name='third_weights',
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                shape=[second_out_shape, third_dense])

        with tf.compat.v1.variable_scope("third_bias", reuse=tf.compat.v1.AUTO_REUSE):
            third_bias = tf.compat.v1.get_variable(                              # Bias Initialization 
                name='third_bias',
                initializer=tf.compat.v1.zeros_initializer,
                shape=[third_dense,])

        third_dense_layer = tf.matmul(second_dropout, third_weights) + third_bias

        third_relu = tf.nn.relu(third_dense_layer)


        third_out_shape = third_relu.shape[1]
        print("third shape", third_out_shape)

        with tf.compat.v1.variable_scope("final_weights", reuse=tf.compat.v1.AUTO_REUSE):

            final_weights = tf.compat.v1.get_variable(                          # Weights Initialization 
                name='final_weights',
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
                shape=[third_out_shape, output_size])

        with tf.compat.v1.variable_scope("final_bias", reuse=tf.compat.v1.AUTO_REUSE):
            final_bias = tf.compat.v1.get_variable(                              # Bias Initialization 
                name='final_bias',
                initializer=tf.compat.v1.zeros_initializer,
                shape=[output_size,])


        final_layer = tf.matmul(third_relu, final_weights) + final_bias

        if training:
            return final_layer, dropout_rate
        else:
            return final_layer

    def train(self, learn_rate, rate_step, dropout_rate, display_step, save_step, batch_size, training_time, eval_step, train_data, Validation_data, init=False):
        """
            Train Model 
            Args
                learn_rate: Should be a list of two values for learning Rate e.g [0.001, 0.0001]
                rate_step: iteration to step into second learning rate
                dropout_rate: percentage of dropout
                display_step: When to display Loss and Validation on training
                save_step: Steps to save Checkpoints, 
                batch_size: batch Size to train at a time
                training_time: Total Training Time
                eval_step: Evaluate Validation Data
                train_data: Provide array of training data with label at the end [-1] index of each data
                Validation_data: Provide array of validation data with label at the end [-1] index of each data 
                init: Default to False for Model initialization at firdt Call of the Model Class
            Returns:
                training and Validation History of Model as Dictionary
        """
        self._save_step = save_step
        self._ground_truth_input = tf.compat.v1.placeholder(
            tf.int64, [None], name='groundtruth_input')

        with tf.compat.v1.name_scope('cross_entropy'):
            self._cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                labels=self._ground_truth_input, logits=self._final_layer)


        learning_rate_input = tf.compat.v1.placeholder(
                tf.float32, [], name='learning_rate_input')

        train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate_input).minimize(self._cross_entropy_mean)

        self._predicted = tf.argmax(input=self._final_layer, axis=1)
        correct_prediction = tf.equal(self._predicted, self._ground_truth_input)
    
        self._evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                                tf.float32))
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

        if self._loaded is False and self._start_step ==0:
            print("global")
            self._global_step = tf.compat.v1.train.get_or_create_global_step()
            tf.compat.v1.global_variables_initializer().run()
            #self._loaded = True

        increment_global_step = tf.compat.v1.assign(self._global_step, self._global_step + 1)
      
        if init is False:
            tf.io.write_graph(self._sess.graph_def, self._model_dir, "model"+ '.pbtxt')
        
        

        if training_time <= self._start_step and self._loaded:
            print(f"Checkpoint Loaded has been trained to {self._start_step},\
                New Trainig starts from {self._start_step}, Please increase Training_time to train model")

        if init is False:
            if tf.config.list_physical_devices('GPU'):
                strategy = tf.distribute.MirroredStrategy()
            else:  # use default strategy
                strategy = tf.distribute.get_strategy() 

            with strategy.scope():
                history = {
                    "categorical_accuracy":[],
                    "loss": [],
                    "val_categorical_accuracy":[],
                    "val_loss":[]}
                learning_rate = learn_rate[0]
                for training_step in xrange(self._start_step, training_time):
                    if training_step == int(rate_step):
                        learning_rate = learn_rate[1]

                    x_train, y_train = self.get_next_batch(batch_size, train_data)
                    train_accuracy, cross_entropy_value, _, _ = self._sess.run(
                        
                        [
                            self._evaluation_step,
                            self._cross_entropy_mean,
                            train_step,
                            increment_global_step,
                        ],
                        feed_dict={
                            self._input: x_train,
                            self._ground_truth_input: y_train,
                            learning_rate_input: learning_rate,
                            self._dropout: dropout_rate
                        })
                    
                    if training_step % int(display_step) ==0:
                        print(
                            'Step #%d: learning rate %f, accuracy %.1f%%, cross entropy %f' %
                            (training_step, learning_rate, train_accuracy * 100,
                            cross_entropy_value))
                        history["categorical_accuracy"].append(train_accuracy)
                        history["loss"].append(cross_entropy_value)

                    if training_step % int(eval_step) ==0:
                        x_val, y_val = self.get_next_batch(batch_size*4, Validation_data)
                        validation_accuracy, val_crossentropy_value = self._sess.run(
                                [
                                    self._evaluation_step, 
                                    self._cross_entropy_mean
                                
                                ],
                                feed_dict={
                                    self._input: x_val,
                                    self._ground_truth_input: y_val,
                                    self._dropout: 0.0
                                })

                        history["val_categorical_accuracy"].append(validation_accuracy)
                        history["val_loss"].append(val_crossentropy_value)

                    
                        print('Step %d: Validation accuracy = %.1f%% (Val Size=%d), Validation loss = %f' %
                                    (training_step, validation_accuracy * 100, batch_size*4, val_crossentropy_value))

                    if (training_step% int(save_step) ==0) or (training_step == training_time-1):
                        path_to_save = os.path.join(self._model_dir, "model_checkpoint" + '.ckpt')
                        saver.save(self._sess, path_to_save, global_step=training_step)
                        self._start_step = self._global_step.eval(session=self._sess)

            return history


    def get_next_batch(self, batch_size, dataset):
        """
        Get Next Batch Size from Dataset
        
        Args:
            Batch_size: Number of Data from the set
            dataset: Dataset to get data from
        returns:
            data: [batch_size, input_size]
            labels: [batch_size, 1]
        """
        np.random.shuffle(dataset)
        
        data = dataset[:batch_size, :-1]
        labels = dataset[:batch_size, -1]

        return data, labels

    def evaluate(self, input_data, labels, verbose= 1):
        """
            Evaluate Data

            Args:
                Input_data: Data points to evaluate
                labels: labels of data
            return:
                validation accuracy
                validation loss
        """

        validation_accuracy, val_crossentropy_value = self._sess.run(
                        [self._evaluation_step, self._cross_entropy_mean],
                        feed_dict={
                            self._input: input_data,
                            self._ground_truth_input: labels,
                            self._dropout: 0.0
                        })
        if verbose:
            print('Validation accuracy = %.1f%%, Validation loss = %f' %
                                (validation_accuracy * 100, val_crossentropy_value))
            
        return validation_accuracy, val_crossentropy_value


    def predict(self, input_data):
        """
        Predict Data input and return Label 

        """

        predicted = self._sess.run(
            [self._final_layer],
            feed_dict = {
                self._input: input_data,
                self._dropout: 0.0

            }
        )
        
        return predicted


    def load_checkpoint(self, path=0):
        """
        Load from checkpoint Path
        """
        
        if path==0:
            try:
                last = int(self._start_step//self._save_step)*self._save_step
                path = os.path.join(self._model_dir, "model_checkpoint" + '.ckpt-'+str(last))
            except:
                print("Checkpoint Path does not Exist, pass path as Arguiment or train for a number of epochs")
                return
        
        #assert os.file.exists(path), "Path does not exist"

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        
        saver.restore(self._sess, path)
        self._start_step = self._global_step.eval(session=self._sess)
        self._loaded = True
        return True

    def save_pb_model(self, save_path, checkpoint_path=None,):

        """
            Save Model For Inference
        """
        input_vector = tf.compat.v1.placeholder(tf.float32, shape=(None, self._input_size))
        
        input_1d = tf.reshape(input_vector, shape=(-1, self._input_size))
        

        softmax_layer = self._build(self._input_size, self._first_dense, self._second_dense, 
                            self._third_dense, self._output_size, training=False, input_1d=input_1d)


        
        output = tf.nn.softmax(softmax_layer, name='labels_softmax')
        
        if checkpoint_path is None: # Should load from last saved checkpoint
            self.load_checkpoint()
        else:
            self.load_checkpoint(path=checkpoint_path)

        build = tf.compat.v1.saved_model.builder.SavedModelBuilder(save_path)
        info_inputs = {
            'input': tf.compat.v1.saved_model.utils.build_tensor_info(input_1d)
        }
        info_outputs = {
            'predictions': tf.compat.v1.saved_model.utils.build_tensor_info(output)
        }
        signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs=info_inputs,
                outputs=info_outputs,
                method_name=tf.compat.v1.saved_model.signature_constants
                .PREDICT_METHOD_NAME))
        build.add_meta_graph_and_variables(
            self._sess,
            [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.compat.v1.saved_model.signature_constants
                .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature,
            },
        )
        build.save()


        

