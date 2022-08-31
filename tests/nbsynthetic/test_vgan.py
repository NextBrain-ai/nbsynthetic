from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.test import TestCase
import numpy as np


class DenseLayerTest(TestCase):
  
  def setUp(self):
      super(DenseLayerTest, self).setUp()
      self.my_dense = tf.keras.layers.Dense(2)
      self.my_dense.build((2, 2))

  def testDenseLayerOutput(self):
    #Maths: activation(dot(input, kernel) + bias)
      self.my_dense.set_weights(
          [        
          np.array([[1, 0],
                    [2, 3]]),#kernel
          np.array([0.5, 0]) #bias
           ]
           )
      input = np.array([[1, 2],
                        [2, 3]])
      output = self.my_dense(input)
      #expected_output = np.dot(
      #    np.array([[1,2], [2,3]]), 
      #    np.array([[1,0],[2,3]])
      #) + np.array([0.5,0]) 
      expected_output = np.array([[5.5, 6.],
                                  [8.5, 9]])

      self.assertAllEqual(expected_output, output)

      """
      Script for running the test in python:
      
      import sys
      import tensorflow as tf
      from test_vgan import DenseLayerTest

      sys.argv = sys.argv[:1]
      tf.test.main()
      """