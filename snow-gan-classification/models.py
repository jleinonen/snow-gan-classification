import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Reshape
from tensorflow.keras.layers import UpSampling2D, Dense, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
from tensorflow.keras.layers import GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from norm import AdaIN, SNConv2D, SNDense, BatchStd


def conv_block(channels, conv_size=(3,3),
    norm=None, stride=1, activation=True, padding='valid'):

    Conv = SNConv2D if norm=="spectral" else Conv2D

    def block(x):
        if norm=="instance":
            x = PixelwiseInstanceNormalization()(x)
        elif norm=="batch":
            x = BatchNormalization(momentum=0.8, scale=False)(x)
        if activation:
            x = LeakyReLU(0.2)(x)
        x = Conv(channels, conv_size, padding=padding,
            strides=(stride,stride))(x)
        return x

    return block


def res_block(channels, conv_size=(3,3), stride=1, norm=None):

    def block(x):
        in_channels = int(x.shape[-1])
        x_in = x
        if (stride > 1):
            x_in = AveragePooling2D(pool_size=(stride,stride))(x_in)
        if (channels != in_channels):
            x_in = conv_block(channels, conv_size=(1,1), stride=1, 
                activation=False)(x_in)

        x = conv_block(channels, conv_size=conv_size, stride=stride,
            padding='same', norm=norm)(x)
        x = conv_block(channels, conv_size=conv_size, stride=1,
            padding='same', norm=norm)(x)

        x = Add()([x,x_in])

        return x

    return block


def dense_block(channels, norm=None, activation=True):

    def block(x):
        
        if norm=="instance":
            x = InstanceNormalization(scale=False)(x)
        elif norm=="batch":
            x = BatchNormalization(momentum=0.8, scale=False)(x)
        if activation:
            x = LeakyReLU(0.2)(x)
        x = SNDense(channels)(x) if norm=="spectral" else Dense(channels)(x)
        return x

    return block


def stylegan_subblock(channels):
    def block(x, noise, style_weights):
        scale = SNDense(channels, kernel_initializer='random_normal', bias_initializer='zeros')(style_weights)
        bias = SNDense(channels, kernel_initializer='random_normal', bias_initializer='zeros')(style_weights)
        x = AdaIN()([x,scale,bias])
        if noise is not None:
            scaled_noise = SNConv2D(channels, kernel_size=(1,1), use_bias=False,
                kernel_initializer='random_normal')(noise)
            x = Add()([x,scaled_noise])
        x = LeakyReLU(0.2)(x)
        x = SNConv2D(channels, kernel_size=(3,3), padding='same')(x)
        return x
    return block


def stylegan_block(channels, upscale=True):
    def block(x, noise, style_weights):
        in_channels = int(x.shape[-1])
        if upscale:
            x = UpSampling2D()(x)
        x_in = x
        if (channels != in_channels):
            x_in = SNConv2D(channels, kernel_size=(1,1))(x_in)

        x = stylegan_subblock(channels)(x_in, noise[0], style_weights)
        x = stylegan_subblock(channels)(x, noise[1], style_weights)
        x = Add()([x,x_in])
        return x
    return block


def stylegan_generator(latent_vars=8):
    noise_shapes = [
        #(512-latent_vars,),
        #(4,4,1), (4,4,1),
        (8,8,1), (8,8,1),
        (16,16,1), (16,16,1),
        (32,32,1), (32,32,1),
        (64,64,1), (64,64,1),
        (128,128,1), (128,128,1),
    ]

    latent_in = Input(shape=(latent_vars,), name="gen_latent_in")
    noise_in = [Input(shape=s, name="gen_noise_in_{}".format(i+1)) for 
        (i,s) in enumerate(noise_shapes)]
    latents = latent_in #Concatenate()([latent_in,noise_in[0]])
    inputs = [latent_in] + noise_in

    w = latents

    initial_shape = (4,4,16)
    x = SNDense(np.prod(initial_shape), use_bias=False)(w)
    x = Reshape(initial_shape)(x)

    x = stylegan_block(512)(x, noise_in[0:2], w)
    x = stylegan_block(256)(x, noise_in[2:4], w)
    x = stylegan_block(128)(x, noise_in[4:6], w)
    x = stylegan_block(64)(x, noise_in[6:8], w)
    x = stylegan_block(64)(x, noise_in[8:10], w)

    img_out = SNConv2D(1, kernel_size=(1,1), padding='same', 
        activation='linear')(x)

    gen = Model(inputs=inputs, outputs=img_out)

    return (gen, noise_shapes)


def resnet_infogan_discriminator(latent_vars=8, size=128):
    img_in = Input(shape=(size,size,1), name="disc_img_in")
    
    x = res_block(64,norm="spectral")(img_in)
    x = res_block(64,stride=2,norm="spectral")(x)
    x = res_block(64,stride=2,norm="spectral")(x)
    x = res_block(128,stride=2,norm="spectral")(x)
    x = res_block(256,stride=2,norm="spectral")(x)
    x = res_block(512,stride=2,norm="spectral")(x)
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = Concatenate()([x_avg,x_max])

    l = dense_block(64, norm="spectral")(x)
    l = dense_block(64, norm="spectral")(l)
    latent_out = SNDense(latent_vars, activation="linear")(l)

    batch_std = BatchStd()(x)
    sx = Concatenate()([x, batch_std])
    sx = dense_block(256, norm="spectral")(sx)
    sx = SNDense(1, activation='linear')(sx)

    inputs = [img_in]
    out = [sx, latent_out]

    model = Model(inputs=inputs, outputs=out, name="disc")
    return model
