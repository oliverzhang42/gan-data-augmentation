# Imports

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import *
from jinja2.compiler import generate

class CGAN(object):
    def __init__(self, encoding_size, neurons_discriminator, neurons_generator,
                 batch_size, results_dir, checkpoint_dir, logs_dir, dataset_name, epochs):
        
        # Discriminator and Generator Architectures Respectively
        self.dis = neurons_discriminator
        self.gen = neurons_generator
        
        # Encoding Size (encoding_size x 1) vector
        self.encoding = encoding_size
        
        self.batch_size = batch_size
        
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        
        self.epochs = epochs
        
        self.z_dim = 64
        
        if dataset_name == "mnist" or "fashion-mnist":
            self.data_X, self.data_y = load_mnist(dataset_name)
            self.image_shape = [28, 28, 1]
            self.label_size = 10
        
        else:
            raise NotImplementedError
        
        assert os.path.isdir(results_dir), "results_dir has to exist"
        #assert os.path.isdir(checkpoint_dir), "checkpoint_dir has to exist"
        #assert os.path.isdir(logs_dir), "logs_dir has to exist"
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        
        self.build_model()
        
        self.sess = tf.Session()
        
    def generator(self, labels, noise, is_training=True, reuse=False):
        
        with tf.variable_scope("generator", reuse=reuse):
            net = concat([labels, noise], axis=1)
        
            for i in range(len(self.dis)):
                net = dense(net, self.dis[i], scope="g_dense_{}".format(i), is_training=is_training)
                net = bn(net, scope="g_bn_{}".format(i), is_training=is_training)
                net = lrelu(net)
        
            net = dense(net, 7*7*128, scope="g_dense_{}".format(len(self.dis) + 1), is_training=is_training)
            net = bn(net, scope="g_bn_{}".format(len(self.dis) + 1), is_training=is_training)
            net = lrelu(net)
        
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
        
            net = deconv2d(net, output_shape = [self.batch_size, 14, 14, 64], scope="g_deconv_1", is_training=is_training)
            net = deconv2d(net, output_shape = [self.batch_size, 28, 28, 1], scope="g_deconv_2", is_training=is_training)
        
            return sigmoid(net)
    
    def discriminator(self, images, labels, is_training=True, reuse=False):
        input_shape = images.get_shape()
        assert len(input_shape) == 4, "The input shape must have four dimensions"
            
        with tf.variable_scope("discriminator", reuse=reuse):
            net = images
        
            for i in range(len(self.gen)):
                net = conv2d(net, self.gen[i], scope="d_conv_{}".format(i), is_training=is_training)
                net = bn(net, scope="d_bn_{}".format(i), is_training=is_training)
                net = lrelu(net)
        
            net = tf.reshape(net, [self.batch_size, -1])
            net = concat([net, labels], axis=1)
        
            net = dense(net, 1024, scope="d_dense_{}".format(1), is_training=is_training)
            net = bn(net, scope="d_bn_{}".format(len(self.dis) + 1), is_training=is_training)
            net = lrelu(net)
            net = dense(net, 1, scope="d_dense_{}".format(2), is_training=is_training)
        
            return sigmoid(net)
        
    def build_model(self):
        image_dims = self.image_shape
        bs = self.batch_size
        
        #images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims)
        
        #labels
        self.y = tf.placeholder(tf.float32, [bs, self.label_size])
        
        #noise
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim])
        
        """Loss Function"""
        # Calculating Answers
        real_answers = self.discriminator(self.inputs, self.y, reuse=False)
        
        fake_images = self.generator(self.y, self.z, reuse=False)
        fake_answers = self.discriminator(fake_images, self.y, reuse=True)
        
        # Discriminator Loss
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_answers, labels=tf.ones_like(real_answers)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_answers, labels=tf.zeros_like(fake_answers)))
        
        self.d_loss = d_loss_fake + d_loss_real
        
        # Generator Loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_answers, labels=tf.ones_like(fake_answers)))
        
        """Training"""
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)
        
        # For Testing
        self.fake_images = self.generator(self.y, self.z, is_training=False, reuse=True)
        
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        
    def train(self):
        bs = self.batch_size
        
        # The number of batches
        num_batches = len(self.data_X) // bs
        
        with self.sess.as_default():
            # initialize variables
            tf.global_variables_initializer().run()
        
        for epoch in range(self.epochs):
            for idx in range(num_batches):
                batch_images = self.data_X[bs*idx:bs*(idx+1)]
                batch_labels = self.data_y[bs*idx:bs*(idx+1)]
            
                batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                
                # Update Discriminator
                
                _, d_loss, = self.sess.run([self.d_optim, self.d_loss], 
                              feed_dict={self.inputs: batch_images, self.y: batch_labels,
                              self.z: batch_noise})
                
                # Update Generator
                _, g_loss, = self.sess.run([self.g_optim, self.g_loss], 
                              feed_dict={self.inputs: batch_images, self.y: batch_labels,
                              self.z: batch_noise})
                
                if idx % 100 == 0:
                    print("Epoch {} Batch {}".format(epoch, idx))
                    print("Discriminator:")
                    print(d_loss)
                    print("Generator:")
                    print(g_loss)
        
        batch_images = self.data_X[0:bs]
        batch_labels = self.data_y[0:bs]
            
        batch_noise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        
        generated_images = self.sess.run(self.fake_images, feed_dict={self.inputs: batch_images, self.y: batch_labels,
                           self.z: batch_noise})
        generated_images = np.array(generated_images)
        generated_images = generated_images.reshape([64, 28, 28])
        
        for i in range(len(generated_images)):
            plt.imshow(generated_images[i], cmap="Greys")
            fname = "Test_{}.png".format(i)
            plt.savefig(os.path.join(self.results_dir, fname))
            plt.clf()