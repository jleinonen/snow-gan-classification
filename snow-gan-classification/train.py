import gc
import numpy as np
import netCDF4
import gan
import data
import models
import plots
import utils


def setup_gan(data_files, stored_gan=None,
    steps_per_epoch=None, latent_vars=8,
    batch_size=32, lr_disc=0.0001, lr_gen=0.0001,
    batch_seed=None, latent_seed=None, noise_seed=None):

    ds_len = 0
    for fn in data_files:
        with netCDF4.Dataset(fn, 'r') as ds:
            ds_len += ds["images"].shape[0]
            img_shape = ds["images"].shape

    images = np.empty((ds_len,)+img_shape[1:], dtype=np.uint8)

    img_ind = 0
    for fn in data_files:
        with netCDF4.Dataset(fn, 'r') as ds:
            N_ds = ds["images"].shape[0]
            images[img_ind:img_ind+N_ds,...] = np.array(
                ds["images"][:], copy=False)
            img_ind += N_ds

    batch_gen = data.BatchGeneratorGAN(images, batch_size=batch_size,
        random_seed=batch_seed)
    img_size = images.shape[1]

    (gen, noise_shapes) = models.stylegan_generator(latent_vars)
    disc = models.resnet_infogan_discriminator(latent_vars, size=img_size)
    wgan = gan.InfoGAN(gen, disc, lr_gen=lr_gen, lr_disc=lr_disc)

    latent_shapes = [(latent_vars,)]
    latent_gen = data.NoiseGenerator(latent_shapes,
        batch_size=batch_gen.batch_size, random_seed=latent_seed)
    noise_shapes = utils.input_shapes(gen, "gen_noise_in")
    noise_gen = data.NoiseGenerator(noise_shapes,
        batch_size=batch_gen.batch_size, random_seed=noise_seed)

    if steps_per_epoch is None:
        steps_per_epoch = batch_gen.N//batch_gen.batch_size

    gc.collect()

    return (wgan, batch_gen, latent_gen, noise_gen, steps_per_epoch)


def train_gan(wgan, batch_gen, latent_gen, noise_gen, steps_per_epoch, num_epochs,
    training_ratio=1):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        wgan.train(batch_gen, latent_gen, steps_per_epoch,
            training_ratio=training_ratio)
        plots.progress_report(wgan.disc, wgan.gen, batch_gen, latent_gen, noise_gen,
            out_fn="../figures/progress.pdf")

    return wgan
