export default class Script {

    constructor(opts={}) {
      
      this.Dis = opts.Dis;
      this.Gen = opts.Gen;
      this.Info = opts.Info;

      this.xTrain = null;
      this.yTrain = null;
      this.numClasses = null;
      this.imageSize = null;
  
      this.batchSize = 64;
      this.latentSize = 100;

      this.softOne = 0.95;
      this.learningRate = 0.0002;
      this.adamBeta1 = 0.5;

      this.statement = `from matplotlib import pyplot
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import randn, randint
from numpy import zeros, ones, expand_dims
from tensorflow.keras.utils import plot_model
from keras.datasets.fashion_mnist import load_data
from keras.initializers import RandomNormal, GlorotNormal
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout, Embedding, Activation, Concatenate\n\n`;
    }
    
    downloadTextFile() {
        this.buildDiscriminator(this.Dis, this.Info);
        this.buildGenerator(this.Gen, this.Info);
        this.buildGAN(this.Info);
        this.buildUility(this.Info);
        this.buildTrainLoop(this.Info)
        const element = document.createElement('a');
        const file = new Blob([this.statement], { type: 'text/plain' });
        element.href = URL.createObjectURL(file);
        element.download = 'gan_script.py';
        element.click();
    }
    
    async conv2DBlock(inp, depth, k_size, stride, pad, init, act, bn, drop, type) {
        if (inp === false) {
            this.statement = this.statement + `\tfe = Conv2D(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=${init})(fe)\n`;
        }
        else {
            if (type === 'Conditional GAN'){
                this.statement = this.statement + `\t# label input
\tin_label = Input(shape=(1,))
\tli = Embedding(n_classes, 50)(in_label)
\tn_nodes = in_shape[0] * in_shape[1]
\tli = Dense(n_nodes)(li)
\tli = Reshape((in_shape[0], in_shape[1], 1))(li)
\t# image input
\tin_image = Input(shape=in_shape)
\tmerge = Concatenate()([in_image, li])
`
                this.statement = this.statement + `\tfe = Conv2D(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=${init})(merge)\n`;
        
            }
            else {
                this.statement = this.statement + `\tin_image = Input(shape=in_shape)\n`
                this.statement = this.statement + `\tfe = Conv2D(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=${init})(in_image)\n`
            };
        }
        if (bn.toLowerCase() === 'yes') {
            this.statement = this.statement + `\tfe = BatchNormalization()(fe)\n`;
        }
        if (act === 'leakyrelu') {
            this.statement = this.statement + `\tfe = LeakyReLU(alpha=0.2)(fe)\n`;
        }
        else {
            this.statement = this.statement + `\tfe = Activation('${act}')(fe)\n`;
        }
        if (drop !== 0) this.statement = this.statement + `\tfe = Dropout(${drop})(fe)\n`;
    }

    async conv2DTransposeBlock(inp, depth, k_size, stride, pad, init, act, bn, drop) {
        if (inp === false) {
            if (init === 'randomNormal'){
                this.statement = this.statement + `\tgen = Conv2DTranspose(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=RandomNormal(mean=0.0, stddev=0.1))(gen)\n`;
            }
            else {
                this.statement = this.statement + `\tgen = Conv2DTranspose(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=${init})(gen)\n`;
            }
        }
        else {
            if (init === 'randomNormal') {
                this.statement = this.statement + `\tgen = Conv2DTranspose(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=RandomNormal(mean=0.0, stddev=0.1))(merge)\n`;
            }
            else {
                this.statement = this.statement + `\tgen = Conv2DTranspose(${depth}, (${k_size}, ${k_size}), strides=(${stride}, ${stride}), padding='${pad}', kernel_initializer=${init})(merge)\n`;
            }
        }
        if (bn.toLowerCase() === 'yes') {
            this.statement = this.statement + `\tgen = BatchNormalization()(gen)\n`;
        }
        if (act === 'leakyrelu') {
            this.statement = this.statement + `\tgen = LeakyReLU(alpha=0.2)(gen)\n`;
        }
        else {
            this.statement = this.statement + `\tgen = Activation('${act}')(gen)\n`;
        }
        if (drop !== 0) this.statement = this.statement + `\tgen = Dropout(${drop})(gen)\n`;
    }

    buildDiscriminator(Dis, Info) {

        const info = Info[0];

        this.statement = this.statement + `def define_discriminator(in_shape=(28, 28, 1), n_classes=10):\n`;
        let init;
        if (info.init === 'randomNormal') {
            init = 'RandomNormal(stddev=0.02)';
        }
        else if (info.init === 'glorotNormal') {
            init = 'GlorotNormal()';
        }
        else {
            init = info.init;
        }
        for (const key in Dis) {
          if (Dis.hasOwnProperty(key)) {
  
            const layer = Dis[key];
            if (key == 0) {
                // First convolution layer
                this.conv2DBlock(true, layer.depth, layer.kernel_size, layer.stride, layer.padding, init, layer.activation.toLowerCase(), layer.batch_normalisation, layer.dropout, info.type);
            }
            else {
                this.conv2DBlock(false, layer.depth, layer.kernel_size, layer.stride, layer.padding, init, layer.activation.toLowerCase(), layer.batch_normalisation, layer.dropout);
            }
          }
        }

        this.statement = this.statement + `\tfe = Flatten()(fe)\n`
        if (info.type === "Semi-Supervised GAN" || info.type === "Auxiliary Classifier GAN") {
            this.statement = this.statement + `\t# real/fake output
\tout1 = Dense(1, activation='sigmoid')(fe)\n
\t# class label output
\tout2 = Dense(n_classes, activation='softmax')(fe)\n
\tmodel = Model(in_image, [out1, out2])
\topt = Adam(learning_rate=0.0002, beta_1=0.5)
\tmodel.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
\treturn model\n\n`;
        }

        else if (info.type === "Conditional GAN") {
            this.statement = this.statement + `\t# real/fake output
\tout1 = Dense(1, activation='sigmoid')(fe)\n
\tmodel = Model([in_image, in_label], out1)
\topt = Adam(learning_rate=0.0002, beta_1=0.5)
\tmodel.compile(loss=['binary_crossentropy'], optimizer=opt)
\treturn model\n\n`;
        }
    
        else {
            this.statement = this.statement + `\t# real/fake output
\tout1 = Dense(1, activation='sigmoid')(fe)\n
\tmodel = Model(in_image, out1)
\topt = Adam(learning_rate=0.0002, beta_1=0.5)
\tmodel.compile(loss=['binary_crossentropy'], optimizer=opt)
\treturn model\n\n`;
        }
    }

    buildGenerator(Gen, Info) {

        const info = Info[0];

        this.statement = this.statement + `def define_generator(latent_dim=100, n_classes=10):\n`;
        let init;
        if (info.init === 'randomNormal') {
            init = 'RandomNormal(stddev=0.02)';
        }
        else if (info.init === 'glorotNormal') {
            init = 'GlorotNormal()';
        }
        else {
            init = info.init;
        }
        if (info.type === "Conditional GAN" || info.type === "Auxiliary Classifier GAN") {
            this.statement = this.statement + `\t# label input
\tin_label = Input(shape=(1,))\n
\t# embedding for categorical input
\tli = Embedding(n_classes, 50)(in_label)\n
\t# linear multiplication
\tn_nodes = ${info.reshape_h} * ${info.reshape_w}\n
\tli = Dense(n_nodes, kernel_initializer=${init})(li)
\t# reshape to additional channel
\tli = Reshape((${info.reshape_h}, ${info.reshape_w}, 1))(li)\n
\t# image generator input
\tin_lat = Input(shape=(latent_dim,))\n
\t# foundation for 7x7 image
\tn_nodes = ${info.reshape_d} * ${info.reshape_h} * ${info.reshape_w}
\tgen = Dense(n_nodes, kernel_initializer=${init})(in_lat)
\tgen = Activation('relu')(gen)
\tgen = Reshape((${info.reshape_h}, ${info.reshape_w}, ${info.reshape_d}))(gen)\n
\t# merge image gen and label input
\tmerge = Concatenate()([gen, li])\n\n`;
        }

        else {
            this.statement = this.statement + `\t# image generator input
\tin_lat = Input(shape=(latent_dim,))\n
\t# foundation for 7x7 image
\tn_nodes = ${info.reshape_d} * ${info.reshape_h} * ${info.reshape_w}
\tgen = Dense(n_nodes, kernel_initializer=${init})(in_lat)
\tgen = Activation('relu')(gen)
\tmerge = Reshape((${info.reshape_h}, ${info.reshape_w}, ${info.reshape_d}))(gen)\n\n`;
        }
        
        for (const key in Gen) {
          if (Gen.hasOwnProperty(key)) {
  
            const layer = Gen[key];
            if (key == 0) {
                // First convolution layer
                this.conv2DTransposeBlock(true, layer.depth, layer.kernel_size, layer.stride, layer.padding, init, layer.activation.toLowerCase(), layer.batch_normalisation, layer.dropout);
            }
            else {
                this.conv2DTransposeBlock(false, layer.depth, layer.kernel_size, layer.stride, layer.padding, init, layer.activation.toLowerCase(), layer.batch_normalisation, layer.dropout);
            }
          }
        }
        if (info.type === "Conditional GAN" || info.type === "Auxiliary Classifier GAN") {
            this.statement = this.statement + `\tmodel = Model([in_lat, in_label], gen)
\treturn model\n\n`;
        }
    
        else {
            this.statement = this.statement + `\tmodel = Model(in_lat, gen)
\treturn model\n\n`;
        }
    }
  
    buildGAN(Info) {

        const info = Info[0];
        this.statement = this.statement + `def define_gan(g_model, d_model):\n`;
        if (info.type === "GAN" || info.type === 'gan') {
            this.statement = this.statement + `\t# make weights in the discriminator not trainable
\tfor layer in d_model.layers:
\t\tif not isinstance(layer, BatchNormalization):
\t\t\tlayer.trainable = False
\t# connect the outputs of the generator to the inputs of the discriminator
\tgan_output = d_model(g_model.output)
\t# define gan model as taking noise and label and outputting real/fake and label outputs
\tmodel = Model(g_model.input, gan_output)
\t# compile model
\topt = Adam(learning_rate=0.0002, beta_1=0.5)
\tmodel.compile(loss=['binary_crossentropy'], optimizer=opt)
\treturn model\n\n`;
            }
        else if (info.type === "Conditional GAN") {
            this.statement = this.statement + `\t# make weights in the discriminator not trainable
\tfor layer in d_model.layers:
\t\tif not isinstance(layer, BatchNormalization):
\t\t\tlayer.trainable = False
\tgen_noise, gen_label = g_model.input
\t# connect the outputs of the generator to the inputs of the discriminator
\tgan_output = d_model([g_model.output, gen_label])
\t# define gan model as taking noise and label and outputting real/fake and label outputs
\tmodel = Model(g_model.input, gan_output)
\t# compile model
\topt = Adam(learning_rate=0.0002, beta_1=0.5)
\tmodel.compile(loss=['binary_crossentropy'], optimizer=opt)
\treturn model\n\n`;
        }
        else {
        this.statement = this.statement + `\t# make weights in the discriminator not trainable
\tfor layer in d_model.layers:
\t\tif not isinstance(layer, BatchNormalization):
\t\t\tlayer.trainable = False
\t# connect the outputs of the generator to the inputs of the discriminator
\tgan_output = d_model(g_model.output)
\t# define gan model as taking noise and label and outputting real/fake and label outputs
\tmodel = Model(g_model.input, gan_output)
\t# compile model
\topt = Adam(learning_rate=0.0002, beta_1=0.5)
\tmodel.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
\treturn model\n\n`;
        }
    }

    buildUility(Info) {
        const info = Info[0];
        this.statement = this.statement + `\n\t# load images
def load_real_samples():
\t# load dataset
\t(trainX, trainy), (_, _) = load_data()
\t# expand to 3d, e.g. add channels
\tX = expand_dims(trainX, axis=-1)
\t# convert from ints to floats
\tX = X.astype('float32')
\t# scale from [0,255] to [-1,1]
\tX = (X - 127.5) / 127.5
\tprint(X.shape, trainy.shape)
\treturn [X, trainy]
        
# select real samples
def generate_real_samples(dataset, n_samples):
\t# split into images and labels
\timages, labels = dataset
\t# choose random instances
\tix = randint(0, images.shape[0], n_samples)
\t# select images and labels
\tX, labels = images[ix], labels[ix]
\t# generate class labels
\ty = ones((n_samples, 1))
\treturn [X, labels], y
        
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
\t# generate points in the latent space
\tx_input = randn(latent_dim * n_samples)
\t# reshape into a batch of inputs for the network
\tz_input = x_input.reshape(n_samples, latent_dim)
\t# generate labels
\tlabels = randint(0, n_classes, n_samples)
\treturn [z_input, labels]\n`;
        if (info.type === 'Conditional GAN' || info.type === 'Auxiliary Classifier GAN') {
            this.statement = this.statement + `# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
\t# generate points in latent space
\tz_input, labels_input = generate_latent_points(latent_dim, n_samples)
\t# predict outputs
\timages = generator.predict([z_input, labels_input], verbose=0)
\t# create class labels
\ty = zeros((n_samples, 1))
\treturn [images, labels_input], y`;
        }
        else {
            this.statement = this.statement + `# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
\t# generate points in latent space
\tz_input, labels_input = generate_latent_points(latent_dim, n_samples)
\t# predict outputs
\timages = generator.predict(z_input, verbose=0)
\t# create class labels
\ty = zeros((n_samples, 1))
\treturn [images, labels_input], y\n`;
        }
        this.statement = this.statement + `# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
\t# prepare fake examples
\t[X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
\t# scale from [-1,1] to [0,1]
\tX = (X + 1) / 2.0
\t# plot images
\tfor i in range(100):
\t\t# define subplot
\t\tpyplot.subplot(10, 10, 1 + i)
\t\t# turn off axis
\t\tpyplot.axis('off')
\t\t# plot raw pixel data
\t\tpyplot.imshow(X[i, :, :, 0], cmap='gray_r')
\t# save plot to file
\tfilename1 = 'generated_plot_%d.png' % (step)
\tpyplot.savefig(filename1)
\tpyplot.close()
\t# save the generator model
\tfilename2 = 'generator_%d.h5' % (step)
\tg_model.save(filename2)
\tprint('>Saved: %s and %s' % (filename1, filename2))\n\n`;
    }

    buildTrainLoop(Info) {
        
        const info = Info[0];
        this.statement = this.statement + `# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=${info.epoch}, n_batch=64):
\t# calculate the number of batches per training epoch
\tbat_per_epo = int(dataset[0].shape[0] / n_batch)
\t# calculate the number of training iterations
\tn_steps = bat_per_epo * n_epochs
\t# calculate the size of half a batch of samples
\thalf_batch = int(n_batch / 2)
\t# manually enumerate epochs
\tfor i in range(n_steps):\n`;
        this.statement = this.statement + `\t\t# get randomly selected 'real' samples\n`;
        if (info.type === 'GAN' || info.type === 'gan') {
            this.statement = this.statement + `\t\t[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
\t\t# update discriminator model weights
\t\td_r1 = d_model.train_on_batch(X_real, y_real)
\t\t# generate 'fake' examples
\t\t[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
\t\t# update discriminator model weights
\t\td_f1 = d_model.train_on_batch(X_fake, y_fake)
\t\t# prepare points in latent space as input for the generator
\t\t[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
\t\t# create inverted labels for the fake samples
\t\ty_gan = ones((n_batch, 1))
\t\t# update the generator via the discriminator's error
\t\tg_1 = gan_model.train_on_batch(z_input, y_gan)
\t\t# summarize loss on this batch
\t\tprint('>%d, dr[%.3f], df[%.3f], g[%.3f]' % (i+1, d_r1, d_f1, g_1))`;
        }
        else if (info.type === 'Conditional GAN') {
            this.statement = this.statement + `\t\t[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
\t\t# update discriminator model weights
\t\td_r1 = d_model.train_on_batch([X_real, labels_real], y_real)
\t\t# generate 'fake' examples
\t\t[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
\t\t# update discriminator model weights
\t\td_f1 = d_model.train_on_batch([X_fake, labels_fake], y_fake)
\t\t# prepare points in latent space as input for the generator
\t\t[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
\t\t# create inverted labels for the fake samples
\t\ty_gan = ones((n_batch, 1))
\t\t# update the generator via the discriminator's error
\t\tg_1 = gan_model.train_on_batch([z_input, z_labels], y_gan)
\t\t# summarize loss on this batch
\t\tprint('>%d, dr[%.3f], df[%.3f], g[%.3f]' % (i+1, d_r1, d_f1, g_1))`;
        }
        else if (info.type === 'Semi-supervised GAN') {
            this.statement = this.statement + `\t\t[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
\t\t# update discriminator model weights
\t\t_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
\t\t# generate 'fake' examples
\t\t[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
\t\t# update discriminator model weights
\t\t_,d_f1,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
\t\t# prepare points in latent space as input for the generator
\t\t[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
\t\t# create inverted labels for the fake samples
\t\ty_gan = ones((n_batch, 1))
\t\t# update the generator via the discriminator's error
\t\t_,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, z_labels])
\t\t# summarize loss on this batch
\t\tprint('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1, d_r2, d_f1, d_f2, g_1, g_2))`;
        }
        else {
            this.statement = this.statement + `\t\t[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
\t\t# update discriminator model weights
\t\t_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
\t\t# generate 'fake' examples
\t\t[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
\t\t# update discriminator model weights
\t\t_,d_f1,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
\t\t# prepare points in latent space as input for the generator
\t\t[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
\t\t# create inverted labels for the fake samples
\t\ty_gan = ones((n_batch, 1))
\t\t# update the generator via the discriminator's error
\t\t_,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
\t\t# summarize loss on this batch
\t\tprint('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1, d_r2, d_f1, d_f2, g_1, g_2))`;
        }
                
        this.statement = this.statement + `\t\t# evaluate the model performance every 'epoch'
\t\tif (i+1) % (bat_per_epo * 10) == 0:
\t\t\tsummarize_performance((i+1) // (bat_per_epo * 10), g_model, latent_dim)
\t\tif (i+1) % bat_per_epo == 0:
\t\t\tprint('====== Finished Epoch', (i+1) // bat_per_epo, '======')
        
# size of the latent space
latent_dim = 100

# create the discriminator
discriminator = define_discriminator()
# plot the model
plot_model(discriminator, to_file='discriminator.png', show_shapes=True)

# create the generator
generator = define_generator(latent_dim)
# plot the model
plot_model(generator, to_file='generator.png', show_shapes=True)

# create the gan
gan_model = define_gan(generator, discriminator)

# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)`;
    }
}