import argparse
import gc

from matplotlib import pyplot as plt
import numpy as np

import cluster
import data
import eval
import plots
import train


def training(model_load_name="",
    model_save_name="../models/masc_infogan_combined",
    batch_size=64, batches_per_epoch=500):
    '''
    Given enough time, this should replicate the GAN training.
    The train_gan function will create a figure called progress.pdf
    in the figures directory, this figure can be used to monitor the
    progress of the training.

    The default model_target_name is set to masc_infogan so you don't
    accidentally overwrite the pre-trained models.
    '''
    (infogan, batch_gen, latent_gen, noise_gen, steps_per_epoch) = \
        train.setup_gan(["../data/masc_davos.nc","../data/masc_apres3.nc"],
        batch_size=batch_size)
    if model_load_name:
        infogan.load(infogan.filenames_from_root(model_load_name))
    while True:
        train.train_gan(infogan, batch_gen, latent_gen, noise_gen,
            batches_per_epoch, 1)
        infogan.save(model_save_name)


def latents(model_name="../models/masc_infogan_combined", 
    latents_fn="../data/masc_latents.nc",
    latent_dist_fn="../data/masc_latent_dist.nc"):
    '''
    Computes the latent codes for the entire dataset for the trained GAN.
    This will take quite a while for the latent-code distributions so it
    should probably be run on a GPU.
    '''

    (infogan, batch_gen, latent_gen, noise_gen, steps_per_epoch) = \
        train.setup_gan(["../data/masc_davos.nc","../data/masc_apres3.nc"],
        batch_size=64)
    infogan.load(infogan.filenames_from_root(model_name))

    eval.latents_all(infogan.disc, batch_gen, out_file=latents_fn)
    eval.latent_dist_all(infogan.disc, batch_gen, out_file=latent_dist_fn)


def experiments(model_name="../models/masc_infogan_combined",
    latents_fn="../data/masc_latent_dist_combined.nc"):
    (latents, ind) = cluster.sample_latents(
        latents_fn, random_seed=1000)
    '''
    Creates the plots and computes the key evaluation numbers reported in 
    the paper. This should be runnable on a CPU given a pre-trained model,
    although it will take some time.
    '''

    # number of clusters
    (K, cost, min_medoids) = cluster.cluster_cost(latents, ind, K_max=20)
    np.save("../data/cluster_losses_combined.npy", (K, cost, min_medoids))

    kmed_16 = cluster.KMedoids(latents,
        metric=cluster.distribution_distance, num_medoids=16)
    kmed_16.medoid_ind = min_medoids[15]
    kmed_16.rearrange_medoids()
    kmed_16.medoid_ind = kmed_16.medoid_ind[[
        4, 0, 1, 3, 2, 15, 13, 14, 11, 12, 9, 10, 5, 7, 6, 8, 
    ]] # cosmetic rearrangement
    (K_16, costs_16) = cluster.hierarchy_cost(kmed_16)
    (branches_16,joins_16) = cluster.cluster_hierarchy(kmed_16)

    kmed_6 = cluster.KMedoids(latents,
        metric=cluster.distribution_distance, num_medoids=6)
    kmed_6.medoid_ind = min_medoids[5]
    kmed_6.rearrange_medoids()
    (K_6, costs_6) = cluster.hierarchy_cost(kmed_6)
    (branches_6,joins_6) = cluster.cluster_hierarchy(kmed_6)

    plots.cluster_number(K, cost, costs_16, costs_6)
    plt.savefig("../figures/Kmedoids_loss.pdf", bbox_inches='tight');
    plt.close()

    # cluster_samples
    plots.cluster_samples(kmed_16, ind, random_seed=1001)
    plt.savefig("../figures/class_samples_16.pdf", bbox_inches='tight');
    plt.close()

    plots.cluster_samples(kmed_6, ind, random_seed=1002)
    plt.savefig("../figures/class_samples_6.pdf", bbox_inches='tight');
    plt.close()

    # cluster distance matrix
    plots.cluster_distance_matrix(kmed_16)
    plt.savefig("../figures/class_distance_matrix.pdf", bbox_inches='tight')
    plt.close()

    # class membership matrix
    plots.class_membership_matrix(kmed_16)
    plt.savefig("../figures/membership_matrix.pdf", bbox_inches='tight')
    plt.close()

    # class statistics matrix
    plots.class_statistics_matrix(kmed_16)
    plt.savefig("../figures/statistics_matrix.pdf", bbox_inches='tight')
    plt.close()

    # GAN samples
    (infogan, batch_gen, latent_gen, noise_gen, steps_per_epoch) = \
        train.setup_gan(["../data/masc_davos.nc","../data/masc_apres3.nc"],
            batch_size=8, latent_seed=1003, noise_seed=1004, batch_seed=1005)
    infogan.load(infogan.filenames_from_root(model_name))
    plots.sample_images(infogan.disc, infogan.gen, batch_gen, latent_gen,
        noise_gen, num_samples=8)
    plt.savefig("../figures/sample_images.pdf", bbox_inches='tight')
    plt.close()

    latent_gen = data.NoiseGenerator([(8,)],
        batch_size=batch_gen.batch_size, random_seed=1040)
    noise_gen = data.NoiseGenerator(noise_gen.noise_shapes,
        batch_size=batch_gen.batch_size, random_seed=1007)
    plots.latent_variation(infogan.gen, noise_gen, latent_gen)
    plt.savefig("../figures/latent_variation.pdf", bbox_inches='tight')
    plt.close()

    gc.collect()

    # Distribution distance / SED comparison
    cam_index = eval.group_images_by_cam([
        "../data/masc_davos.nc",
        "../data/masc_apres3.nc"
    ])
    (d, dm, dr, d_dist, dm_dist, dr_dist) = eval.evaluate_distance(
        "../data/masc_latent_combined.nc",
        "../data/masc_latent_dist_combined.nc",
        cam_index, random_seed=1008)
    print("Median SED (pairs) = {:.3f}".format(np.median(d)))
    print("Median SED (all) = {:.3f}".format(np.median(dm)))
    print("Median distance rank for SED = {:.3f}%".format(np.median(dr)*100))
    print("Median Bhattacharyya distance (pairs) = {:.3f}".format(
        np.median(d_dist)))
    print("Median Bhattacharyya distance (all) = {:.3f}".format(
        np.median(dm_dist)))
    print("Median distance rank for Bhattacharyya distance = {:.3f}%".format(
        np.median(dr_dist)*100))

    gc.collect()

    # SSIM
    (median_ssim, mean_ssim, std_ssim) = eval.average_ssim(
        infogan.gen, infogan.disc, batch_gen, noise_gen, 
        num_batches=128, verbose=True)
    print("Median SSIM = {:.3f}".format(median_ssim))
    print("Mean SSIM = {:.3f}".format(mean_ssim))
    print("SSIM st. dev. = {:.3f}".format(std_ssim))

    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
        help="'train', 'latents' or 'experiments'")
    parser.add_argument('--model_name', type=str,
        default="",
        help="Name of the model to load")
    parser.add_argument('--model_save_name', type=str, default="",
        help="Name of the model to save (if mode is 'train')")
    parser.add_argument('--latents_file', type=str, 
        default="../data/masc_latents.nc",
        help="File to save the computed latent variables (if mode is 'latents')")
    parser.add_argument('--latent_dist_file', type=str, 
        default="../data/masc_latent_dist.nc",
        help="File to save the computed latent variable distribution (if mode is 'latents')")

    args = parser.parse_args()
    mode = args.mode
    model_name = args.model_name
    model_save_name = args.model_save_name
    latents_fn = args.latents_file
    latent_dist_fn = args.latent_dist_file

    if mode == "train":
        training(model_load_name=model_name, model_save_name=model_save_name)
    elif mode == "latents":
        latents(model_name=model_name, latents_fn=latents_fn,
            latent_dist_fn=latent_dist_fn)
    elif mode == "experiments":
        experiments(model_name=model_name)
