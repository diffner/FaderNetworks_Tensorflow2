import tensorflow as tf
from Code.training import *
from Code.model import AEloss_full, Dloss
from Code.loader import normalize_images
import datetime
import numpy as np
import time

class Training():

    def __init__(self, ae_model, disc_model, params, training_data, test_data=None, logdir = "anything"):
        self.ae_model = ae_model
        self.disc_model = disc_model

        self.epochs = params['epochs']
        self.ae_optimizer = params['ae_optimizor']
        self.disc_optimizer = params['disc_optimizor']
        self.lambda_ae = params['lambda_ae']
        self.lambda_e = params['lambda_e']
        self.scaled_lambda = 0
        self.logdir = logdir
        self.training_data = training_data
        self.test_data = test_data
        
    
    def train_models(self):
        """ Train the AE and Discriminator for the number of specified epochs

        - Get z from encoder
        - Train discriminator with z and y
        - Get loss for enc, dec
        - train_step([enc, dec]], loss, optimizer_disc)
        """
        self.disc_graph_logdir = self.logdir + '/graphs/disc/'
        self.disc_graph_writer = tf.summary.create_file_writer(self.disc_graph_logdir)
        self.ae_graph_logdir = self.logdir + '/graphs/AE'
        self.ae_graph_writer = tf.summary.create_file_writer(self.ae_graph_logdir)
        self.train_log_dir = self.logdir + '/gradient_tape/train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.log_ae_loss = tf.keras.metrics.Mean('AE_loss', dtype=tf.float32)
        self.log_re_loss = tf.keras.metrics.Mean('RE', dtype=tf.float32)
        self.log_d_loss = tf.keras.metrics.Mean('Disc_loss', dtype=tf.float32)

        #keep track on the number of iterations (needed to scale lambda)
        nr_iteration = 156250
      
        for epoch in range(500, 500 + self.epochs):
            start = time.time()
            print()
            print(epoch + 1)
            print()
            for step, batch in enumerate(self.training_data):
                X_batch = normalize_images(tf.cast(batch[0], 'float32'))
                Y_batch = batch[1]
                Z_batch = self.ae_model.encode(X_batch)

                if nr_iteration == 0:
                    tf.summary.trace_on(graph=True, profiler=True)
                
                self.train_step_disc(Z_batch, Y_batch)
                # Call only one tf.function when tracing.
                
                if nr_iteration == 0:
                    with self.disc_graph_writer.as_default():
                        tf.summary.trace_export(
                            name="disc_loss_graph",
                            step=0,
                            profiler_outdir=self.disc_graph_logdir)

                    tf.summary.trace_on(graph=True, profiler=True)
                
                #ADD LAMBDA SCHEDULE ACCORDING TO OUR EXPERIMENTS AND EPOCH LENGTH
                self.scale_lambda(self.lambda_e, nr_iteration)
                self.train_step_ae(X_batch, Y_batch, Z_batch)

                if nr_iteration == 0:
                    with self.ae_graph_writer.as_default():
                        tf.summary.trace_export(
                            name="AE_loss_graph",
                            step=0,
                            profiler_outdir=self.ae_graph_logdir)

                nr_iteration += 1
            end = time.time()
            print("Epoch " + str(epoch + 1) + " takes " + str(end - start))
            with self.train_summary_writer.as_default():
              tf.summary.scalar('Auto Encoder Adversarial Loss', self.log_ae_loss.result(), step=epoch)
              tf.summary.scalar('Reconstruction Error', self.log_re_loss.result(), step=epoch)
              tf.summary.scalar('Discriminator Loss', self.log_d_loss.result(), step=epoch)

            #GET LOSSES HERE
            template = 'Epoch {}, AE Adversarial Loss: {}, RE: {}, Discriminator Loss: {}'
            print (template.format(epoch+1,
                                  self.log_ae_loss.result(),
                                  self.log_re_loss.result(),
                                  self.log_d_loss.result()))
            self.log_ae_loss.reset_states()
            self.log_re_loss.reset_states()
            self.log_d_loss.reset_states()

            #print("epoch: " + str(epoch) + " AE LOSS: " + str(ae_loss.numpy()) + " D LOSS: " + str(d_loss.numpy()))
            self.ae_model.save_weights(self.logdir + "/Models/AEModelCkpt.h5")
            self.disc_model.save_weights(self.logdir + "/Models/DiscModelCkpt.h5")
            if(epoch%20 == 0):
                self.ae_model.save_weights(self.logdir + "/Models/AEModelCkpt_epoch_" + str(epoch) + ".h5")
                self.disc_model.save_weights(self.logdir + "/Models/DiscModelCkpt_epoch_" + str(epoch) + ".h5")
    
    def scale_lambda(self, lmbd, nr_iterations, lmbd_schedule = 500000):
        """
        Linearly increases lambda up to its actual value after lmbd_schedule number of iterations
        """
        self.scaled_lambda = lmbd * float(min(lmbd_schedule,nr_iterations)) / lmbd_schedule 

    @tf.function
    def train_step_disc(self, z, y):
        """Executes one training step for discriminator model.
        Computes loss and update gradients
        """
        with tf.GradientTape() as tape:
            loss = Dloss(self.disc_model, z, y)
        gradients = tape.gradient(loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(
            (g, v) for (g,v) in zip(gradients, self.disc_model.trainable_variables))
        self.log_d_loss(loss)
        
    @tf.function
    def train_step_ae(self, x, y, z):
        """Executes one training step for AE model.
        Computes loss and update gradients
        """
        with tf.GradientTape() as tape:
            loss = AEloss_full(self.ae_model, x, y, self.disc_model, self.lambda_ae, self.scaled_lambda)
        fool = Dloss(self.disc_model, z, 1-y)
        RE = (loss - self.scaled_lambda * fool)/self.lambda_ae
        gradients = tape.gradient(loss, self.ae_model.trainable_variables)
        self.ae_optimizer.apply_gradients(
            (g, v) for (g,v) in zip(gradients, self.ae_model.trainable_variables))
        self.log_ae_loss(loss)
        self.log_re_loss(RE)
