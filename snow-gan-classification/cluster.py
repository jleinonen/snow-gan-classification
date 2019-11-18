import dask
import numpy as np
import netCDF4


def bhattacharyya_distance(mean1, cov1, mean2, cov2):
    '''Bhattacharyya distance for multivariate Gaussians

    mean1 and cov1 may contain multiple samples, in which case
    the first dimension should be the sample dimension
    ''' 

    cov = 0.5*(cov1+cov2)
    mean_diff = mean1-mean2
    mean_term = (mean_diff*np.linalg.solve(cov, mean_diff)).sum(axis=-1) / 8
    cov_term = 0.5 * np.log(
        np.linalg.det(cov)/np.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2))
    )
    return mean_term+cov_term


def distribution_distance(y1,y2):
    if (y1.ndim == 1) and (y2.ndim == 1):
        m1 = y1[:8] 
        c1 = y1[8:].reshape((8,8)) 
        m2 = y2[:8] 
        c2 = y2[8:].reshape((8,8))
    elif y1.ndim == 2:
        m1 = y1[:,:8] 
        c1 = y1[:,8:].reshape((y1.shape[0],8,8)) 
        m2 = y2[:8] 
        c2 = y2[8:].reshape((8,8))
    elif y2.ndim == 2:
        m1 = y2[:,:8] 
        c1 = y2[:,8:].reshape((y2.shape[0],8,8)) 
        m2 = y1[:8] 
        c2 = y1[8:].reshape((8,8))
    else:
        raise ValueError("Invalid dimensions.")

    return bhattacharyya_distance(m1,c1,m2,c2)


class KMedoids(object):
    def __init__(self, X, metric, num_medoids=2, random_seed=None,
        distance_matrix=None, medoid_ind=None):
        self.X = X
        self.metric = metric
        self.num_medoids = num_medoids
        self.N = X.shape[0]
        self.random_seed = random_seed
        self.prng = np.random.RandomState(seed=random_seed)
        self.distances = {}
        if medoid_ind is None:
            self.init_medoids()
        else:
            self.medoid_ind = medoid_ind
        if distance_matrix is None:
            self.init_distance_matrix()
        else:
            self.distance_matrix = distance_matrix

    def init_medoids(self):
        self.medoid_ind = self.prng.choice(self.N, self.num_medoids)

    def init_distance_matrix(self):
        self.distance_matrix = np.zeros((self.N,self.N))
        for i in range(self.N):
            self.distance_matrix[i,i+1:] = self.metric(
                self.X[i+1:,:], self.X[i,:])
        self.distance_matrix += self.distance_matrix.T

    def get_distance(self, i1, i2):
        return self.distance_matrix[i1,i2]

    def get_cost(self, medoid_ind=None):
        if medoid_ind is None:
            medoid_ind = self.medoid_ind
        cost = 0.0
        dm = self.distance_matrix[medoid_ind,:]
        cost = dm.min(axis=0).mean()

        return cost

    def iterate(self, verbose=True):
        cost = self.get_cost()
        if verbose:
            print(cost)
        for (j,i_m) in enumerate(self.medoid_ind):
            if verbose:
                print("Cluster {}/{}".format(j+1,self.num_medoids))
            points = self.prng.choice(self.N, self.N, replace=False)
            for i_p in points:
                if i_p in self.medoid_ind:
                    continue
                medoid_ind = self.medoid_ind.copy()
                medoid_ind[j] = i_p
                new_cost = self.get_cost(medoid_ind=medoid_ind)
                if new_cost < cost:
                    self.medoid_ind = medoid_ind
                    cost = new_cost
                    if verbose:
                        print(cost)

        return cost

    def fit(self):
        old_cost = np.inf
        while True:
            cost = self.iterate()
            if cost >= old_cost:
                break
            old_cost = cost
        return cost

    def labels(self):
        labels = np.empty(self.N)
        for i_p in range(self.N):
            dist_nearest = np.inf
            for (j,i_m) in enumerate(self.medoid_ind):
                dist = self.get_distance(i_m,i_p)
                if dist < dist_nearest:
                    label = j
                    dist_nearest = dist
            labels[i_p] = label
        return labels.astype(int)

    def medoid_distance_matrix(self):
        return self.distance_matrix[self.medoid_ind,:][:,self.medoid_ind]

    def rearrange_medoids(self):
        (branches, joins) = cluster_hierarchy(self)
        order = np.array([b for b in iterate_branch(branches)])
        self.medoid_ind = self.medoid_ind[order]

    def copy(self):
        c = KMedoids(self.X.copy(), self.metric, num_medoids=self.num_medoids,
            random_seed=self.random_seed, medoid_ind=self.medoid_ind.copy(),
            distance_matrix=self.distance_matrix.copy())
        return c


def iterate_branch(b):
    if len(b)>1:
        yield from iterate_branch(b[0])
        yield from iterate_branch(b[1])
    else:
        yield b[0]


def cluster_hierarchy(kmed):

    branches = [[l] for l in np.arange(kmed.num_medoids)]
    joins = []

    dist = kmed.medoid_distance_matrix().copy()
    def branch_distance(b1,b2):
        d = 0.0
        n = 0
        for i in iterate_branch(b1):
            for j in iterate_branch(b2):
                d += dist[i,j]
                n += 1
        return d/n

    while len(branches) > 1:
        d = np.inf
        for (i,b1) in enumerate(branches):
            for (j,b2) in list(enumerate(branches))[i+1:]:
                d_new = branch_distance(b1,b2)
                if d_new < d:
                    d = d_new
                    (min_i,min_j) = (i,j)

        join_i = next(iterate_branch(branches[min_i]))
        join_j = next(iterate_branch(branches[min_j]))
        branches[min_i] = [branches[min_i], branches[min_j]]
        del branches[min_j]
        joins.append((join_i,join_j))

    return (branches[0], joins)


def sample_latents(latents_file, N_latents=2048, ind=None, random_seed=None):
    with netCDF4.Dataset(latents_file, 'r') as ds:
        latent_mean = ds["latent_mean"][:]
        latent_cov = ds["latent_cov"][:]

    latent = np.concatenate([
        np.array(latent_mean, copy=False), 
        np.array(latent_cov, copy=False).reshape(
            (latent_cov.shape[0],latent_cov.shape[1]*latent_cov.shape[2]))
    ], axis=-1)

    if ind is None:
        prng = np.random.RandomState(seed=random_seed)
        ind = prng.choice(latent.shape[0], 2048)
    latent_choice = latent[ind,...]

    return (latent_choice, ind)


def cluster_cost(latent_choice, ind, K_max=20, cost=None, 
    max_tries=8, num_workers=4):

    def cost_for_K(K):
        cost = np.inf

        kmed = KMedoids(latent_choice, distribution_distance, num_medoids=K,
            random_seed=5678+K)

        tries = 0
        while tries < max_tries:
            kmed.init_medoids()

            c = np.inf
            improving = True
            while improving:
                kmed.iterate(verbose=False)
                c_new = kmed.get_cost()
                improving = (c_new < c)
                if improving:
                    c = c_new

            if c < cost:
                cost = c
                min_medoids = kmed.medoid_ind
                tries = 0 
            else:
                tries += 1

        print("Cost for K={}: {:.3f}".format(K,cost))

        return (cost,min_medoids)

    K_range = np.arange(1,K_max+1)
    tasks = [dask.delayed(cost_for_K)(K) for K in K_range]
    results = dask.compute(tasks, scheduler="multiprocessing",
        num_workers=num_workers)[0]
    cost = np.array([r[0] for r in results])
    min_medoids = [np.array(r[1]) for r in results]

    return (K_range, cost, min_medoids)


def hierarchy_cost(kmed):
    kmed = kmed.copy()
    (branches, joins) = cluster_hierarchy(kmed)
    labels = kmed.labels()

    def cost():
        return kmed.distance_matrix[
            np.arange(kmed.N),kmed.medoid_ind[labels]
        ].mean()

    def medoid_for_label(label):
        members = np.arange(kmed.N)[labels==label]
        min_dist = np.inf
        for i in members:
            mean_dist = kmed.distance_matrix[i,members].mean()
            if mean_dist < min_dist:
                min_dist = mean_dist
                medoid = i
        return medoid 

    K = [kmed.num_medoids]
    costs = [cost()]
    for (i,j) in joins:
        labels[labels==j] = i
        kmed.medoid_ind[i] = medoid_for_label(i)
        K.append(K[-1]-1)
        costs.append(cost())

    return (np.array(K), np.array(costs))
