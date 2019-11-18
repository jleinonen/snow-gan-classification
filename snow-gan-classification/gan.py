import json
import gc
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Average, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import generic_utils
import tensorflow as tf
from norm import RandomWeightedAverage
import utils


class InfoGAN(object):

    def __init__(self, gen, disc, load_fn_root=None,
        gradient_penalty_weight=10, lr_disc=0.0001, 
        lr_gen=0.0001):

        if load_fn_root is not None:
            load_files = self.filenames_from_root(load_fn_root)
            with open(load_files["gan_params"]) as f:
                params = json.load(f)
            gradient_penalty_weight = params["gradient_penalty_weight"]
            lr_disc = params["lr_disc"]
            lr_gen = params["lr_gen"]

        self.gen = gen
        self.disc = disc
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.build()

        if load_fn_root is not None:
            self.load(load_files)


    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5",
            "gan_params": root+"-gan_params.json"
        }
        return fn


    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])
        
        self.disc.trainable = False
        self.gen_trainer._make_train_function()
        utils.load_opt_weights(self.gen_trainer,
            load_files["gen_opt_weights"])
        self.disc.trainable = True
        
        self.gen.trainable = False
        self.disc_trainer._make_train_function()
        utils.load_opt_weights(self.disc_trainer,
            load_files["disc_opt_weights"])
        self.gen.trainable = True


    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        utils.save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        utils.save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])
        params = {
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "lr_disc": self.lr_disc,
            "lr_gen": self.lr_gen
        }
        with open(paths["gan_params"], 'w') as f:
            json.dump(params, f)


    def build(self):

        # find shapes for inputs
        noise_shapes = utils.input_shapes(self.gen, "gen_noise_in")
        latent_shape = utils.input_shapes(self.gen, "gen_latent_in")[0]
        generated_shape = self.gen.output_shape[1:]

        # Create optimizers
        self.opt_disc = Adam(self.lr_disc, beta_1=0.5, beta_2=0.9)
        self.opt_gen = Adam(self.lr_gen, beta_1=0.5, beta_2=0.9)

        # Create generator training network
        self.disc.trainable = False
        latent = Input(shape=latent_shape)
        noise = [standard_normal_noise(s)(latent) for s in noise_shapes]
        img_est = self.gen([latent]+noise)
        (disc_gen, latent_est) = self.disc([img_est, latent])

        self.gen_trainer = Model(inputs=latent, outputs=[disc_gen, latent_est])
        self.gen_trainer.compile(loss=[generator_loss, 'mse'],
            optimizer=self.opt_gen)
        self.disc.trainable = True

        # Create discriminator training network
        self.gen.trainable = False

        latent = Input(shape=latent_shape)
        noise = [standard_normal_noise(s)(latent) for s in noise_shapes]
        img_est = self.gen([latent]+noise)
        
        img = Input(shape=generated_shape)
        img_avg = RandomWeightedAverage()([img, img_est])

        (disc_real, latent_real) = self.disc(img)
        (disc_gen, latent_gen) = self.disc(img_est)
        (disc_avg, latent_avg) = self.disc(img_avg)
        disc_gp = GradientPenalty()([disc_avg, img_avg])

        self.disc_trainer = Model(
            inputs=[img,latent],
            outputs=[disc_real,disc_gen,disc_gp,latent_gen]
        )
        self.disc_trainer.compile(
            loss=[discriminator_loss,discriminator_loss,'mse','mse'], 
            loss_weights=[1.0,1.0,self.gradient_penalty_weight,1.0],
            optimizer=self.opt_disc
        )
        self.gen.trainable = True


    def train(self, batch_gen, latent_gen, num_gen_batches=1, 
        training_ratio=1, show_progress=True):

        disc_target_real = None
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                num_gen_batches*batch_gen.batch_size)

        for k in range(num_gen_batches):
        
            # Train discriminator
            try:
                self.gen.trainable = False
                disc_loss = None
                disc_loss_n = 0
                for rep in range(training_ratio):
                    # generate some real samples
                    (Y_real, cond) = next(batch_gen)
                    latent = latent_gen()

                    if disc_target_real is None: # on the first iteration
                        # run discriminator once just to find the shapes
                        disc_outputs = self.disc_trainer.predict(
                            [Y_real]+latent)
                        disc_target_real = np.ones(disc_outputs[0].shape,
                            dtype=np.float32)
                        disc_target_fake = -disc_target_real
                        gp_target = np.zeros(disc_outputs[2].shape, 
                            dtype=np.float32)
                        del disc_outputs

                    dl = self.disc_trainer.train_on_batch([Y_real]+latent,
                        [disc_target_real, disc_target_fake, gp_target, latent[0]])

                    if disc_loss is None:
                        disc_loss = np.array(dl)
                    else:
                        disc_loss += np.array(dl)
                    disc_loss_n += 1

                    del Y_real, cond

                disc_loss /= disc_loss_n
            finally:
                self.gen.trainable = True

            # Train generator
            try:
                self.disc.trainable = False
                latent = latent_gen()
                gen_loss = self.gen_trainer.train_on_batch(
                    latent, [disc_target_fake, latent[0]])
            finally:
                self.disc.trainable = True

            if show_progress:
                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                for (i,gl) in enumerate(gen_loss):
                    losses.append(("G{}".format(i), gl))
                progbar.add(batch_gen.batch_size, 
                    values=losses)

        gc.collect()


def generator_loss(y_true, y_pred):
    return K.sum(y_true*y_pred, axis=-1)


def discriminator_loss(y_true, y_pred):
    l = y_true*y_pred
    l = tf.maximum(np.float32(1)-l,np.float32(0))
    return K.sum(l,axis=-1)


class GradientPenalty(Layer):

    def call(self, inputs):
        target, wrt = inputs
        grad = K.gradients(target, wrt)[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


def standard_normal_noise(shape):
    def std_noise(dummy_input):
        batch_shape = K.shape(dummy_input)[:1]
        full_shape = K.constant(shape, shape=(len(shape),), dtype=np.int32)
        full_shape = K.concatenate([batch_shape,full_shape])
        return K.random_normal(full_shape, 0, 1)

    return Lambda(std_noise)
