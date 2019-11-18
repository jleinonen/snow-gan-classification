import numpy as np
import netCDF4
import tensorflow as tf
from tensorflow.keras import backend as K

import cluster
import data


def latents_all(disc, batch_gen, out_file=None, batch_size=128, num_latents=8):

    latents = np.zeros((batch_gen.N, num_latents), dtype=np.float32)

    old_batch_size = batch_gen.batch_size
    try:
        batch_gen.batch_size = batch_size

        num_classified = 0
        while num_classified < batch_gen.N:
            ind = batch_gen.next_indices()
            (Y_real, cond) = next(batch_gen)
            Y_real = batch_gen.augment_image_batch(Y_real)
            (d, l) = disc.predict(Y_real)
            latents[ind,:] = l
            num_classified += batch_gen.batch_size
            print("{}/{}".format(num_classified,batch_gen.N))
    finally:
        batch_gen.batch_size = old_batch_size

    if out_file is not None:
        with netCDF4.Dataset(out_file, 'w') as ds:
            dim_samples = ds.createDimension("dim_samples", batch_gen.N)
            dim_latent = ds.createDimension("dim_latent", num_latents)
            var_params = {"zlib": True, "complevel": 9}

            var_latent = ds.createVariable("latent", np.float32, 
                ("dim_samples","dim_latent"), **var_params)
            var_latent[:] = latents

    return latents


def latent_dist_all(disc, batch_gen, out_file=None, 
    batch_size=128, num_latents=8, num_samples=100):

    latent_mean = np.zeros((batch_gen.N, num_latents), dtype=np.float32)
    latent_cov = np.zeros((batch_gen.N, num_latents, num_latents), dtype=np.float32)
    samples = np.zeros((batch_size, num_latents, num_samples))

    old_batch_size = batch_gen.batch_size
    old_augment = batch_gen.augment
    try:
        batch_gen.batch_size = batch_size
        batch_gen.augment = False

        num_classified = 0
        while num_classified < batch_gen.N:
            ind = batch_gen.next_indices()
            (X, Y) = next(batch_gen)
            for k in range(num_samples):
                samples[:,:,k] = disc.predict(
                    batch_gen.augment_image_batch(X))[1]
            latent_mean[ind,:] = samples.mean(axis=-1)
            for (i,j) in enumerate(ind):
                latent_cov[j,:,:] = np.cov(samples[i,:,:])
            num_classified += batch_gen.batch_size
            print("{}/{}".format(num_classified,batch_gen.N))
    finally:
        batch_gen.batch_size = old_batch_size
        batch_gen.augment = old_augment

    if out_file is not None:
        with netCDF4.Dataset(out_file, 'w') as ds:
            dim_samples = ds.createDimension("dim_samples", batch_gen.N)
            dim_latent = ds.createDimension("dim_latent", num_latents)
            var_params = {"zlib": True, "complevel": 9}

            var_latent_mean = ds.createVariable("latent_mean", np.float32, 
                ("dim_samples","dim_latent"), **var_params)
            var_latent_mean[:] = latent_mean
            var_latent_cov = ds.createVariable("latent_cov", np.float32, 
                ("dim_samples","dim_latent","dim_latent"), **var_params)
            var_latent_cov[:] = latent_cov

    return (latent_mean, latent_cov)


def ssim(img1, img2, sess=None):
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)

    return K.get_session().run(tf.image.ssim(img1, img2, 1.0))


def average_ssim(gen, disc, batch_gen, noise_gen, num_batches=128, verbose=False):
    if verbose:
        print("Drawing batches...")
    batches = [next(batch_gen)[0] for i in range(num_batches)]
    if verbose:
        print("Predicting latents...")
    latents = [disc.predict(b)[1] for b in batches]
    if verbose:
        print("Generating images...")
    gen_batches = [gen.predict([l]+noise_gen()) for l in latents]

    if verbose:
        print("Computing SSIM...")
    ssim_batches = [ssim(b,gb) for (b,gb) in zip(batches,gen_batches)]

    ssim_all = np.concatenate(ssim_batches,axis=0)

    return (np.median(ssim_all), ssim_all.mean(), ssim_all.std())


def group_images_by_cam(fn_list):
    names = []
    for fn in fn_list:
        with netCDF4.Dataset(fn, 'r') as ds:
            names.append(ds["names"][:])
    names = np.concatenate(names, axis=0)

    index = {}

    for i in range(names.shape[0]):
        name = data.extract_name(names[i,:])
        parts = name.split("_")
        date = "_".join(parts[:2])
        unique_id = int(parts[-3])
        cam = int(parts[-1])

        key = (date,unique_id)
        if key not in index:
            index[key] = {}
        index[key][cam] = i

    return index


def evaluate_distance(fn_latent, fn_latent_dist, cam_index, n=1024,
    random_seed=None):

    with netCDF4.Dataset(fn_latent) as ds:
        latent = np.array(ds["latent"][:], copy=False)
    with netCDF4.Dataset(fn_latent_dist) as ds:
        latent_mean = np.array(ds["latent_mean"][:], copy=False)
        latent_cov = np.array(ds["latent_cov"][:], copy=False)

    prng = np.random.RandomState(seed=random_seed)

    distance = []
    distance_mean = []
    distance_rank = []
    distribution_distance = []
    distribution_distance_mean = []
    distribution_distance_rank = []

    keys = [k for k in cam_index if len(cam_index[k]) >= 2]
    prng.shuffle(keys)

    for (i,k) in enumerate(keys[:n]):
        cams = list(cam_index[k].keys())
        i0 = cam_index[k][cams[0]]
        i1 = cam_index[k][cams[1]]

        d = ((latent[i1,:]-latent[i0,:])**2).sum()
        dm = ((latent-latent[i0,:])**2).sum(axis=-1)
        dr = np.count_nonzero(d<dm)/dm.shape[0]

        d_dist = cluster.bhattacharyya_distance(
            latent_mean[i1,:], latent_cov[i1,:,:],
            latent_mean[i0,:], latent_cov[i0,:,:],
        )
        dm_dist = cluster.bhattacharyya_distance(
            latent_mean, latent_cov,
            latent_mean[i0,:], latent_cov[i0,:,:],
        )
        dr_dist = np.count_nonzero(d_dist<dm_dist)/dm_dist.shape[0]

        distance.append(d)
        distance_mean.append(dm.mean())
        distance_rank.append(dr)
        distribution_distance.append(d_dist)
        distribution_distance_mean.append(dm_dist.mean())
        distribution_distance_rank.append(dr_dist)

        d_ratio = (np.array(distance)/np.array(distance_mean)).mean()
        d_dist_ratio = (np.array(distribution_distance) / \
            np.array(distribution_distance_mean)).mean()
        d_rank_mean = np.array(distance_rank).mean()
        d_dist_rank_mean = np.array(distribution_distance_rank).mean()

        print("{}: d_ratio={:.3f}, d_dist_ratio={:.3f}, d_rank={:.3f}, d_dist_rank={:.3f}".format(
            i,d_ratio,d_dist_ratio,d_rank_mean,d_dist_rank_mean))

    return (np.array(distance), np.array(distance_mean), np.array(distance_rank),
        np.array(distribution_distance), 
        np.array(distribution_distance_mean),
        np.array(distribution_distance_rank),
        )
