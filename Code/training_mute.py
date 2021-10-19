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

        #keep track on the number of iterations (needed to scale lambda)
        nr_iteration = 0
      
        for epoch in range(self.epochs):
            start = time.time()
            print()
            print(epoch + 1)
            print()
            for step, batch in enumerate(self.training_data):
                X_batch = normalize_images(tf.cast(batch[0], 'float32'))
                Y_batch = batch[1]
                Z_batch = self.ae_model.encode(X_batch)
                
                self.train_step_disc(Z_batch, Y_batch)
                # Call only one tf.function when tracing.
                #ADD LAMBDA SCHEDULE ACCORDING TO OUR EXPERIMENTS AND EPOCH LENGTH
                self.scale_lambda(self.lambda_e, nr_iteration)
                self.train_step_ae(X_batch, Y_batch, Z_batch)

                nr_iteration += 1
            end = time.time()
            print("Epoch " + str(epoch + 1) + " takes " + str(end - start))

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
