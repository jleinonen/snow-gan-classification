import numpy as np
import netCDF4
from scipy.ndimage import rotate, map_coordinates


class BatchGenerator(object):
    
    def __init__(self, images, labels=None, batch_size=32, augment=True,
        random_seed=None, cutout_prob=0.0, translate_margin=0, shuffle=True):

        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)
        self.images = images
        self.labels = labels
        self.augment = augment
        self.N = self.images.shape[0]
        self.next_ind = np.array([], dtype=int)
        self.img_size = self.images.shape[1]
        self.cutout_prob = cutout_prob
        self.translate_margin = translate_margin
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        ind = self.next_indices()
        self.next_ind = self.next_ind[self.batch_size:]

        X = self.images[ind,...]
        X = X.astype(np.float32) / 255.0
        if self.augment:
            X = self.augment_image_batch(X)
        if self.labels is not None:
            Y = self.labels[ind,...]
        else:
            Y = []

        return (X, Y)

    def next_indices(self):
        while len(self.next_ind) < self.batch_size:
            ind = np.arange(self.N, dtype=int)
            if self.shuffle:
                self.prng.shuffle(ind)
            self.next_ind = np.concatenate([self.next_ind, ind])
        return self.next_ind[:self.batch_size]

    def random_flip(self, img):
        # mirror
        fliplr = bool(self.prng.randint(2))
        if fliplr:
            img = np.fliplr(img)
        flipud = bool(self.prng.randint(2))
        if flipud:
            img = np.flipud(img)
        return img


    def random_rotate(self, img, reshape=True):
        # rotate
        rot_angle_deg = self.prng.rand()*360.0
        return rotate(img, rot_angle_deg, reshape=reshape)


    def augment_image(self, image):
        img = image.copy()

        # crop
        (nzi, nzj) = img[:,:,0].nonzero()
        i0 = nzi.min()
        i1 = nzi.max()+1
        j0 = nzj.min()
        j1 = nzj.max()+1
        img = img[i0:i1,j0:j1,...]

        img = self.random_flip(img)
        img = self.random_rotate(img)

        # fit to image size if needed
        if img.shape[0] > self.img_size:
            i0 = img.shape[0]//2 - self.img_size//2
            i1 = i0+self.img_size
            img = img[i0:i1,:,...]
        if img.shape[1] > self.img_size:
            j0 = img.shape[1]//2 - self.img_size//2
            j1 = j0+self.img_size
            img = img[:,j0:j1,...]

        # translate
        tm = self.translate_margin
        i0_range = (tm, self.img_size-img.shape[0]+1-tm)
        j0_range = (tm, self.img_size-img.shape[1]+1-tm)
        if (i0_range[0] < i0_range[1]) and (j0_range[0] < j0_range[1]):
            new_i0 = self.prng.randint(*i0_range)
            new_i1 = new_i0+img.shape[0]
            new_j0 = self.prng.randint(*j0_range)
            new_j1 = new_j0+img.shape[1]

        img_new = np.zeros_like(image)
        img_new[new_i0:new_i1,new_j0:new_j1,:] = img

        return img_new


    def augment_image_batch(self, images):
        images = images.copy()
        for i in range(images.shape[0]):
            images[i,...] = self.augment_image(images[i,...])
        return images


    def label_weights(self):
        (labels, counts) = np.unique(self.labels, return_counts=True)
        weights = counts.sum()/(len(labels)*counts)
        return (labels, weights)


class BatchGeneratorGAN(BatchGenerator):
    def __init__(self, *args, **kwargs):
        super(BatchGeneratorGAN, self).__init__(*args, **kwargs)
        (self.x,self.y) = np.mgrid[:self.img_size,:self.img_size]
        self.middle = self.img_size//2
        self.x -= self.middle
        self.y -= self.middle
        self.hist_eq = HistogramEqualizer()


    def augment_image(self, image):
        theta = self.prng.rand()*(2*np.pi)
        (ct,st) = (np.cos(theta), np.sin(theta))
        (x,y) = (ct*self.x-st*self.y), (st*self.x+ct*self.y)
        zoom = 0.9+self.prng.rand()*0.1
        x *= zoom
        y *= zoom
        if bool(self.prng.randint(2)):
            x = -x
        if bool(self.prng.randint(2)):
            y = -y
        x_shift = (self.prng.rand()-0.5)*8
        y_shift = (self.prng.rand()-0.5)*8
        
        x += x_shift+self.middle
        y += y_shift+self.middle
        
        img = self.hist_eq(image[:,:,0])
        img = map_coordinates(img, (x,y), order=1)
        brightness_mul = 1 + ((self.prng.rand()-0.5) * 0.2)
        img *= brightness_mul
        img.clip(max=1,out=img)
        img[img<(1/255)] = 0

        return img.reshape(img.shape+(1,))


class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=32, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __call__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size,) + shape

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]


class HistogramEqualizer(object):
    def __init__(self, mode=0.1, map_to=0.2):
        self.mode = mode
        self.map_to = map_to

    def __call__(self, images):
        mode = self.mode
        map_to = self.map_to
        low = (images < mode)
        high = (images >= mode) 
        images_eq = images.copy()
        images_eq[low] *= map_to/mode
        images_eq[high] = (images_eq[high]-mode)*((1.0-map_to)/(1.0-mode))+map_to
        return images_eq


def extract_name(name_enc):
    return "".join(b.decode() for b in 
        netCDF4.stringtochar(name_enc)[::4,0])
