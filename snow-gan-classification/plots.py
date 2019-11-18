import imageio
import netCDF4
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors, gridspec

import cluster
import data
import eval


def plot_img(img, ssim=None):
    plt.imshow(img, interpolation='nearest',
        norm=colors.Normalize(0,1), cmap='gray')
    ax = plt.gca()
    if ssim is not None:
        plt.text(0.98, 0.98, "SSIM = {:.3f}".format(ssim), 
            color='w', fontsize=10, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right")
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def plot_images(images, num_cols=10, labels=None):
    N = images.shape[0]
    num_rows = int(np.ceil(N/num_cols))
    fig = plt.figure(figsize=(1.0*num_cols,1.0*num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.02,
        hspace=0.02)
    for (k,img) in enumerate(images):
        plt.subplot(gs[k])
        plot_img(img[:,:,0])
        label = k if (labels is None) else labels[k]
        plt.text(2,2,str(label),color='w',verticalalignment='top')


def progress_report(disc, gen, batch_gen, latent_gen, noise_gen, num_samples=16,
    out_fn=None):

    (Y_real, cond) = next(batch_gen)
    num_samples = min(num_samples, Y_real.shape[0])
    (disc_real, latent) = disc.predict(Y_real)
    Y_fake = gen.predict([latent]+noise_gen())
    Y_rand = gen.predict(latent_gen()+noise_gen())

    ssim = eval.ssim(Y_real, Y_fake)

    fig = plt.figure(figsize=(6,2*num_samples))
    gs = gridspec.GridSpec(num_samples, 3)
    for i in range(num_samples):
        plt.subplot(gs[i,0])
        plot_img(Y_real[i,:,:,0])
        plt.subplot(gs[i,1])
        plot_img(Y_fake[i,:,:,0], ssim=ssim[i])
        plt.subplot(gs[i,2])
        plot_img(Y_rand[i,:,:,0])
            
    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def sample_images(disc, gen, batch_gen, latent_gen, noise_gen, num_samples=8,
    out_fn=None):

    (Y_real, cond) = next(batch_gen)
    num_samples = min(num_samples, Y_real.shape[0])
    (disc_real, latent) = disc.predict(Y_real)
    Y_fake = gen.predict([latent]+noise_gen())
    Y_rand = gen.predict(latent_gen()+noise_gen())

    ssim = eval.ssim(Y_real, Y_fake)

    fig = plt.figure(figsize=(2*num_samples,6))
    gs = gridspec.GridSpec(3, num_samples, hspace=0.05, wspace=0.1)
    for i in range(num_samples):
        plt.subplot(gs[0,i])
        plot_img(Y_real[i,:,:,0])
        if i==0:
            plt.ylabel("Real")
        plt.subplot(gs[1,i])
        plot_img(Y_fake[i,:,:,0], ssim=ssim[i])
        if i==0:
            plt.ylabel("Reconstructed\nfrom latent")
        plt.subplot(gs[2,i])
        plot_img(Y_rand[i,:,:,0])
        if i==0:
            plt.ylabel("Randomly\ngenerated")
            
    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def latent_variation(gen, noise_gen, latent_gen, latent_ind=(2,3)):
    latent_range = np.linspace(-2,2,9)

    fig = plt.figure(figsize=(1.5*len(latent_range),1.5*len(latent_range)))
    gs = gridspec.GridSpec(len(latent_range), len(latent_range),
        hspace=0.02,wspace=0.02)

    latents = []
    latent = latent_gen()[0][0,:]
    for x in latent_range:
        for y in latent_range:
            latent[latent_ind[0]] = x
            latent[latent_ind[1]] = y
            latents.append(latent.copy())

    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = len(latents)
        noise = noise_gen()
    finally:
        noise_gen.batch_size = old_batch_size

    latents = np.stack(latents)
    generated_images = gen.predict([latents]+noise)

    def label(x):
        return "${}{}\\sigma$".format(
            "+" if x > 0 else "",
            x
        )

    k = 0
    for i in range(len(latent_range)):
        for j in range(len(latent_range)):
            plt.subplot(gs[len(latent_range)-1-i,j])
            plot_img(generated_images[k,:,:,0])
            k += 1
            if i==0:
                plt.xlabel(label(latent_range[j]))
            if j==0:
                plt.ylabel(label(latent_range[i]))


def interpolate_latents(x0, x1, num_points=100):
    r0 = np.sqrt((x0**2).sum())
    r1 = np.sqrt((x1**2).sum())
    r = np.linspace(r0, r1, num_points)

    t = np.linspace(0,1,num_points)
    x = x0[None,:]*(1-t[:,None])+x1[None,:]*t[:,None]
    r_x = np.sqrt((x**2).sum(1))
    x *= (r/r_x)[:,None]

    return x


def animation_frames(gen, disc, batch_gen, noise_gen, frame_dir, 
    num_intermediate_points=100, num_flakes=20, norm_latents=True):
    try:
        old_batch_size = batch_gen.batch_size
        batch_gen.batch_size = num_flakes
        (Y_real, cond) = next(batch_gen)
    finally:
        batch_gen.batch_size = old_batch_size

    (fake, latent) = disc.predict(Y_real)
    if norm_latents:
        latent /= latent.std()    

    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = 1
        noise = noise_gen()
    finally:
        noise_gen.batch_size = old_batch_size

    for i in range(len(noise)):
        noise[i] = np.repeat(noise[i], num_intermediate_points, axis=0)

    frame = 0
    for k in range(num_flakes-1):
        l0 = latent[k,:]
        l1 = latent[k+1,:]
        l = interpolate_latents(l0, l1)
        generated_images = gen.predict([l]+noise)
        for i in range(generated_images.shape[0]):
            img = generated_images[i,:,:,0].copy()
            img.clip(0,1,img)
            img = (img*255).round().astype(np.uint8)
            fn = "{}/frame-{:06d}.png".format(frame_dir,frame)
            imageio.imsave(fn, img)
            frame += 1
            

def cluster_samples(kmedoids, img_ind, images_per_cluster=12, 
    image_fn=["../data/masc_davos.nc","../data/masc_apres3.nc"],
    label_order=None, random_seed=None):

    
    def load_image_at_index(i):
        starting_ind = 0
        for fn in image_fn:
            with netCDF4.Dataset(fn, 'r') as ds:
                if i > (starting_ind + ds["images"].shape[0]):
                    starting_ind += ds["images"].shape[0]
                else:
                    return np.array(ds["images"][i-starting_ind,:,:,0],
                        copy=False)/255.0

    labels = kmedoids.labels()
    ind = np.arange(kmedoids.N)
    prng = np.random.RandomState(seed=random_seed)

    num_rows = kmedoids.num_medoids
    num_cols = images_per_cluster
    plt.figure(figsize=(1.5*num_cols,1.5*num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, hspace=0.05, wspace=0.02)

    if label_order is None:
        label_order = range(kmedoids.num_medoids)
    for (i,label) in enumerate(label_order):
        label_ind = prng.choice(ind[labels==label], images_per_cluster-1,
            replace=False)
        label_ind = img_ind[label_ind]
        label_ind = np.hstack((img_ind[kmedoids.medoid_ind[i]], label_ind))

        for j in range(images_per_cluster):
            plt.subplot(gs[i,j])
            if j==0:
                plt.ylabel(str(label+1))
            plot_img(load_image_at_index(label_ind[j]))


def cluster_distance_matrix(kmedoids):
    import seaborn as sns
    d = kmedoids.medoid_distance_matrix()
    plt.figure(figsize=(10,8))
    (i,j) = np.mgrid[:d.shape[0],:d.shape[1]]
    labels = [str(i) for i in range(1,d.shape[0]+1)]
    sns.heatmap(d, annot=True, fmt=".1f",
        xticklabels=labels, yticklabels=labels, square=True)


def class_membership_matrix(kmedoids):
    import seaborn as sns

    with netCDF4.Dataset("../data/masc_davos.nc") as ds:
        label_probs = np.array(ds["label_probs"][:], copy=False)
    labels_davos = label_probs.argmax(axis=-1)
    with netCDF4.Dataset("../data/masc_apres3.nc") as ds:
        label_probs = np.array(ds["label_probs"][:], copy=False)
    labels_apres3 = label_probs.argmax(axis=-1)
    labels = np.concatenate((labels_davos,labels_apres3))

    with netCDF4.Dataset("../data/masc_latent_dist_combined.nc") as ds:
        latent_mean = np.array(ds["latent_mean"][:], copy=False)
        latent_cov = np.array(ds["latent_cov"][:], copy=False)

    X_medoids = kmedoids.X[kmedoids.medoid_ind,:]
    mean_medoids = X_medoids[:,:8]
    cov_medoids = X_medoids[:,8:].reshape((kmedoids.num_medoids,8,8))

    def nearest_medoid(i):
        m = latent_mean[i,:]
        c = latent_cov[i,:,:]
        dist = cluster.bhattacharyya_distance(
            mean_medoids, cov_medoids, m, c)
        return dist.argmin()

    medoids = np.array([nearest_medoid(i)
         for i in range(latent_mean.shape[0])])

    label_bins = np.arange(0, label_probs.shape[1]+1)-0.5
    medoid_bins = np.arange(0, kmedoids.num_medoids+1)-0.5
    (H, lb, mb) = np.histogram2d(labels, medoids, (label_bins, medoid_bins))

    plt.figure(figsize=(10,2.5))
    H_norm = H / H.sum(axis=0)
    H_tot = H.sum(axis=1)
    H_tot /= H_tot.sum()
    H_norm = np.hstack((H_norm, H_tot[:,None]))
    sns.heatmap(H_norm[1:,:]*100, annot=True, fmt=".1f", square=True,
        xticklabels=[str(i) for i in range(1,H.shape[1]+1)]+["Tot"],
        yticklabels=["CC","PC","AG","GR","CPC"],
    )
    plt.title("Class correspondence (%)")

    return H


def class_statistics_matrix(kmedoids):
    import seaborn as sns

    with netCDF4.Dataset("../data/masc_davos.nc") as ds:
        davos_last_ind = ds["images"].shape[0]

    with netCDF4.Dataset("../data/masc_latent_dist_combined.nc") as ds:
        latent_mean = np.array(ds["latent_mean"][:], copy=False)
        latent_cov = np.array(ds["latent_cov"][:], copy=False)

    X_medoids = kmedoids.X[kmedoids.medoid_ind,:]
    mean_medoids = X_medoids[:,:8]
    cov_medoids = X_medoids[:,8:].reshape((kmedoids.num_medoids,8,8))

    def nearest_medoid(i):
        m = latent_mean[i,:]
        c = latent_cov[i,:,:]
        dist = cluster.bhattacharyya_distance(
            mean_medoids, cov_medoids, m, c)
        return dist.argmin()

    medoids = np.array([nearest_medoid(i)
         for i in range(latent_mean.shape[0])])

    medoid_bins = np.arange(0, kmedoids.num_medoids+1)-0.5
    (h_davos, b) = np.histogram(medoids[:davos_last_ind], medoid_bins)
    (h_apres3, b) = np.histogram(medoids[davos_last_ind:], medoid_bins)
    (h_tot, b) = np.histogram(medoids, medoid_bins)
    H = np.vstack([h_davos, h_apres3, h_tot])

    plt.figure(figsize=(10,1.5))
    H_norm = H / H.sum(axis=1)[:,None]
    sns.heatmap(H_norm*100, annot=True, fmt=".1f", square=True,
        xticklabels=[str(i) for i in range(1,H.shape[1]+1)],
        yticklabels=["Davos","APRES3","Total"], vmin=0
    )
    plt.title("Class membership (%)")

    return H


def cluster_number(K, cost_K, cost_h1, cost_h2):
    plt.plot(K, cost_K, linewidth=1.5, label="$K$-medoids")
    plt.plot(np.arange(len(cost_h1),0,-1), cost_h1, '--',
        linewidth=1.5,
        label="Hierarchical from $K={}$".format(len(cost_h1)))
    plt.plot(np.arange(len(cost_h2),0,-1), cost_h2, ':',
        linewidth=1.5,
        label="Hierarchical from $K={}$".format(len(cost_h2)))
    plt.gca().set_xlim((1,20))
    plt.gca().set_ylim((0,1.05*cost_K.max()))
    plt.gca().set_xticks(np.arange(1,21))
    plt.grid(axis='x', linewidth=0.25, alpha=0.5)
    plt.xlabel("Number of medoids $K$")
    plt.ylabel("Cost")
    plt.legend(loc="upper right")
