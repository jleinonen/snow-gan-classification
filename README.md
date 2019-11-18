## GAN-based unsupervised classification of snowflake images

This Python/TensorFlow code demonstrates unsupervised snowflake classification from images obtained with the [Multi-Angle Snowflake Camera](http://www.inscc.utah.edu/~tgarrett/Snowflakes/MASC.html) using a GAN and K-medoids classification. It supports a paper "Unsupervised classification of snowflake images using a generative adversarial network and K-medoids classification" to be submitted to Atmospheric Measurement Techniques and provides all code needed to replicate the results. 

## Instructions

### Requirements
You need a Python 3 environment and the following libraries:
- TensorFlow (not tested with 2.0+)
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Python NetCDF4
- h5py
- imageio
- Dask

A GPU is highly recommended for training, but the experiments with pre-trained models can be run on a CPU as well. 16+ GB of RAM should be enough.

### Data

Download the training datasets [here](https://github.com/jleinonen/snow-gan-classification/releases/download/datasets/masc_combined_datasets.zip) (they are too big to include in the repository). 
Save the `.nc` and `.npy` files in the `data` directory.

If you want to use the pre-trained models, you can download them [here](https://github.com/jleinonen/snow-gan-classification/releases/download/models/masc_infogan_combined.zip).
Save the contents of the zip file in the `models` directory.

### Running the code

The high-level code that runs the training and evaluation needed to replicate the results can be found in `replication.py`. This file has a command line interface (see below), but you could also call the functions within from an iPython terminal or a Jupyter notebook. 

If you want to modify the training code, you should start by following the code flow in `replication.training`.

#### Running the plotting and evaluation

You can evaluate the model and generate the plots shown in the paper using the downloadable datasets and the pre-trained GAN on the command line in the `snow-gan-classification` directory like this:
```bash
python replication.py experiments --model_name=../models/masc_infogan_combined
```
where `model_name` is the name of the model you want to load (use the default for the pre-trained model). For the pre-trained model, this should replicate the results exactly. If you trained the GAN yourself, you probably will get slightly different results.

In practice, you may want to run the experiments one by one by copypasting the code from `replication.experiments` to a terminal.

#### Training the GAN

You can run the training like this:
```bash
python replication.py train --model_save_name=../models/masc_infogan
```
Change the `--model_save_name` parameter to the name of the model you want to save. You can load a pre-existing model at the start of training using the `--model_name` parameter. So, for example, to load the pre-trained model and train it further:
```bash
python replication.py train --model_name=../models/masc_infogan_combined --model_save_name=../models/masc_infogan
```

#### Computing the latent variables

Run the following to compute the latent variables for all snowflakes in the dataset:
```bash
python replication.py latents --model_name=../models/masc_infogan_combined --latents_file=../data/masc_latents.nc --latent_dist_file=../data/masc_latent_dist.nc
```
where the `--latents_file` and `--latent_dist_file` parameters control where the latents are saved.
