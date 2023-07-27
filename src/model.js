import * as tf from '@tensorflow/tfjs';

export default class GAN {

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
  
      this.generator = null;
      this.discriminator = null;
      this.combinedModel = null;
    }
  
    async setTrainingData(opts, i, l) {

      let images, labels;

      if (opts === "Mnist" || opts === "mnist") {
        
        let data = await loadMnistData();
        images = data.images;
        labels = data.labels;
      }

      else if (opts === "Fashion Mnist") {
        let data = await loadFashionMnistData();
        images = data.images;
        labels = data.labels;
      }
      
      else {
        let data = await getData(i, l);
        images = data.images;
        labels = data.labels;
      }
      
      if (images.shape[1] !== images.shape[2]) throw new Error("Images must be square.");
      this.imageSize = images.shape[1];
      this.numClasses = labels.shape[1];
      this.xTrain = images;
      this.yTrain = tf.expandDims(labels.argMax(-1), -1);
      
      const canvas = document.createElement('canvas');
      document.body.appendChild(canvas);

      if(!this.combinedModel) {

        this.discriminator = this.buildDiscriminator(this.Dis, this.Info);
        this.discriminator.summary();

        this.generator = this.buildGenerator(this.latentSize, this.Gen, this.Info);
        this.generator.summary();
  
        this.combinedModel = this.buildCombinedModel(this.latentSize, this.generator, this.discriminator, this.Info);
        
      }
    }

    buildGenerator(latentSize, Gen, Info) {

      tf.util.assert(latentSize > 0 && Number.isInteger(latentSize), `Expected latent-space size to be a positive integer, but got ${latentSize}.`);
      
      const cnn = tf.sequential();

      const info = Info[0];

      cnn.add(tf.layers.dense({units: info.reshape_h * info.reshape_w * info.reshape_d, inputShape: [latentSize], activation: 'relu'}));
      cnn.add(tf.layers.reshape({targetShape: [info.reshape_h, info.reshape_w, info.reshape_d]}));

      for (const key in Gen) {
        if (Gen.hasOwnProperty(key)) {

          const layer = Gen[key];
          if (layer.activation.toLowerCase() === 'leakyrelu') {
            cnn.add(tf.layers.conv2dTranspose({filters: layer.depth, kernelSize: layer.kernel_size, strides: layer.stride, padding: layer.padding.toLowerCase(), kernelInitializer: info.init}));
            cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
          }
          else {
            cnn.add(tf.layers.conv2dTranspose({filters: layer.depth, kernelSize: layer.kernel_size, strides: layer.stride, padding: layer.padding.toLowerCase(), activation: layer.activation.toLowerCase(), kernelInitializer: info.init}));
          }          
          if (layer.batch_nomalisation === "yes" || layer.batch_nomalisation === "Yes") {
            cnn.add(tf.layers.batchNormalization());
          }
          cnn.add(tf.layers.dropout({rate: layer.dropout}))
        }
      }

      // This is the z space commonly referred to in GAN papers.
      const latent = tf.input({shape: [latentSize]});

      if (info.type === "Conditional GAN" || info.type === "Auxiliary Classifier GAN") {
        // The desired label of the generated image, an integer in the interval [0, this.numClasses).
        const imageClass = tf.input({shape: [1]});
    
        // The desired label is converted to a vector of length `latentSize` through embedding lookup.
        const classEmbedding = tf.layers.embedding({inputDim: this.numClasses, outputDim: latentSize, embeddingsInitializer: info.init}).apply(imageClass);
    
        // Hadamard product between z-space and a class conditional embedding.
        const h = tf.layers.multiply().apply([latent, classEmbedding]);
    
        const fakeImage = cnn.apply(h);
        return tf.model({inputs: [latent, imageClass], outputs: fakeImage});
      }

      else {
        const fakeImage = cnn.apply(latent);
        return tf.model({inputs: latent, outputs: fakeImage});
      }
    }
  
    buildDiscriminator(Dis, Info) {

      const info = Info[0];
      const cnn = tf.sequential();

      for (const key in Dis) {
        if (Dis.hasOwnProperty(key)) {

          const layer = Dis[key];
          if (key == 0) {
            // First convolution layer
            if (layer.activation.toLowerCase() === 'leakyrelu') {
              cnn.add(tf.layers.conv2d({filters: layer.depth, kernelSize: layer.kernel_size, strides: layer.stride, padding: layer.padding.toLowerCase(), kernelInitializer: info.init, inputShape: [info.ishape_h, info.ishape_w, info.ishape_d]}));
              cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
            }
            else {
              cnn.add(tf.layers.conv2d({filters: layer.depth, kernelSize: layer.kernel_size, strides: layer.stride, padding: layer.padding.toLowerCase(), activation: layer.activation.toLowerCase(), kernelInitializer: info.init, inputShape: [info.ishape_h, info.ishape_w, info.ishape_d]}));
            }          
            if (layer.batch_nomalisation === "yes" || layer.batch_nomalisation === "Yes") {
              cnn.add(tf.layers.batchNormalization());
            }
            cnn.add(tf.layers.dropout({rate: layer.dropout}))
          }
          else {
            if (layer.activation.toLowerCase() === 'leakyrelu') {
              cnn.add(tf.layers.conv2d({filters: layer.depth, kernelSize: layer.kernel_size, strides: layer.stride, padding: layer.padding.toLowerCase(), kernelInitializer: info.init}));
              cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
            }
            else {
              cnn.add(tf.layers.conv2d({filters: layer.depth, kernelSize: layer.kernel_size, strides: layer.stride, padding: layer.padding.toLowerCase(), activation: layer.activation.toLowerCase(), kernelInitializer: info.init}));
            }          
            if (layer.batch_nomalisation === "yes" || layer.batch_nomalisation === "Yes") {
              cnn.add(tf.layers.batchNormalization());
            }
            cnn.add(tf.layers.dropout({rate: layer.dropout}))
          }
        }
      }

      cnn.add(tf.layers.flatten());

      let features;
      const image = tf.input({shape: [info.ishape_h, info.ishape_w, info.ishape_d]});
      const imageClass = tf.input({shape: [1]});
      if (info.type === "Conditional GAN") {    
        // The desired label is converted to a vector of length `latentSize` through embedding lookup.
        const classEmbedding = tf.layers.embedding({inputDim: this.numClasses, outputDim: 100, embeddingsInitializer: info.init}).apply(imageClass);
        const deClassEmbedding = tf.layers.dense({units: info.ishape_h * info.ishape_w * info.ishape_d, activation: 'relu'}).apply(classEmbedding);
        const reClassEmbedding = tf.layers.reshape({targetShape: [info.ishape_h, info.ishape_w, info.ishape_d]}).apply(deClassEmbedding);
        // Hadamard product between z-space and a class conditional embedding.
        const h = tf.layers.multiply().apply([image, reClassEmbedding]);
    
        features = cnn.apply(h);
      }

      else {
        features = cnn.apply(image);
      }
      const realnessScore = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(features);

      if (info.type === "Semi-Supervised GAN" || info.type === "Auxiliary Classifier GAN") {
        
        // The output for class labels
        const aux = tf.layers.dense({units: this.numClasses, activation: 'softmax'}).apply(features);
    
        let discriminator = tf.model({inputs: image, outputs: [realnessScore, aux]});
  
        discriminator.compile({
          optimizer: tf.train.adam(this.learningRate, this.adamBeta1),
          loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
        });
        return discriminator;
      }
      
      else if (info.type === "Conditional GAN") {
        console.log('ok');
        let discriminator = tf.model({inputs: [image, imageClass], outputs: realnessScore});
  
        discriminator.compile({
          optimizer: tf.train.adam(this.learningRate, this.adamBeta1),
          loss: ['binaryCrossentropy']
        });
        return discriminator;
      }

      else {
        let discriminator = tf.model({inputs: image, outputs: realnessScore});
  
        discriminator.compile({
          optimizer: tf.train.adam(this.learningRate, this.adamBeta1),
          loss: ['binaryCrossentropy']
        });
        return discriminator;
      }
    }
  
    buildCombinedModel(latentSize, generator, discriminator, Info) {
  
      const latent = tf.input({shape: [latentSize]});
      const imageClass = tf.input({shape: [1]});
      const info = Info[0];

      let fakeImage;
      let aux;
      if (info.type === "Conditional GAN" || info.type === "Auxiliary Classifier GAN") {
        fakeImage = generator.apply([latent, imageClass]);
      }
      else {
        fakeImage = generator.apply(latent);
      }
      
      discriminator.trainable = false;

      if (info.type === "Semi-Supervised GAN" || info.type === "Auxiliary Classifier GAN") {
        [fakeImage, aux] = discriminator.apply(fakeImage);
      }
      else if (info.type === "Conditional GAN") {
        fakeImage = discriminator.apply([fakeImage, imageClass]);
      }
      else {
        fakeImage = discriminator.apply(fakeImage);
      }

      if (info.type === "GAN" || info.type === "gan") {
        const combined = tf.model({inputs: latent, outputs: fakeImage});
        const optimizer = tf.train.adam(this.learningRate, this.adamBeta1);
        combined.compile({optimizer, loss: ['binaryCrossentropy']});
        return combined;
      }
      else if (info.type === "Conditional GAN") {
        const combined = tf.model({inputs: [latent, imageClass], outputs: fakeImage});
        const optimizer = tf.train.adam(this.learningRate, this.adamBeta1);
        combined.compile({optimizer, loss: ['binaryCrossentropy']});
        return combined;
      }
      else if (info.type === "Semi-Supervised GAN") {
        const combined = tf.model({inputs: latent, outputs: [fakeImage, aux]});
        const optimizer = tf.train.adam(this.learningRate, this.adamBeta1);
        combined.compile({optimizer, loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']});
        return combined;
      }
      else {
        const combined = tf.model({inputs: [latent, imageClass], outputs: [fakeImage, aux]});
        const optimizer = tf.train.adam(this.learningRate, this.adamBeta1);
        combined.compile({optimizer, loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']});
        return combined;
      }
    }
  
    async trainDiscriminatorOneStep(xTrain, batchStart, batchSize) {
  
      const imageBatch = xTrain.slice(batchStart, batchSize);
      let zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1);
      const generatedImages = this.generator.predict([zVectors], {batchSize: batchSize});
  
      const x = tf.concat([imageBatch, generatedImages], 0);
      const y = tf.tidy(() => {
        return tf.concat([tf.ones([batchSize, 1]).mul(this.softOne), tf.zeros([batchSize, 1])]);
      });
  
      const losses = await this.discriminator.trainOnBatch(x, [y]);
      tf.dispose([x, y]);
      return losses;
  
    }
    
    async trainCombinedModelOneStep(batchSize) {
  
      const zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1); // <-- noise

      const trick = tf.ones([batchSize, 1]).mul(this.softOne);
  
      const losses = await this.combinedModel.trainOnBatch([zVectors], [trick]);
      tf.dispose([zVectors, trick]);
      return losses;
  
    }

    async trainCDiscriminatorOneStep(xTrain, yTrain, batchStart, batchSize) {
  
      const imageBatch = xTrain.slice(batchStart, batchSize);
      const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');
  
      let zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1);
      let sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');
  
      const generatedImages = this.generator.predict([zVectors, sampledLabels], {batchSize: batchSize});
  
      const x = tf.concat([imageBatch, generatedImages], 0);
  
      const y = tf.tidy(() => {
        return tf.concat([tf.ones([batchSize, 1]).mul(this.softOne), tf.zeros([batchSize, 1])]);
      });
      const auxY = tf.concat([labelBatch, sampledLabels], 0);
  
      const losses = await this.discriminator.trainOnBatch([x, auxY], [y]);
      tf.dispose([x, y,  auxY]);
      return losses;
  
    }
  
    async trainCCombinedModelOneStep(batchSize) {
  
      const zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1); // <-- noise
      const sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');
      const trick = tf.ones([batchSize, 1]).mul(this.softOne);
  
      const losses = await this.combinedModel.trainOnBatch([zVectors, sampledLabels], [trick]);
      tf.dispose([zVectors, sampledLabels, trick]);
      return losses;
  
    }

    async trainSDiscriminatorOneStep(xTrain, yTrain, batchStart, batchSize) {
  
      const imageBatch = xTrain.slice(batchStart, batchSize);
      const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');
  
      let zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1);
      let sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');
  
      const generatedImages = this.generator.predict([zVectors], {batchSize: batchSize});
  
      const x = tf.concat([imageBatch, generatedImages], 0);
  
      const y = tf.tidy(() => {
        return tf.concat([tf.ones([batchSize, 1]).mul(this.softOne), tf.zeros([batchSize, 1])]);
      });
  
      const auxY = tf.concat([labelBatch, sampledLabels], 0);
  
      const losses = await this.discriminator.trainOnBatch(x, [y, auxY]);
      tf.dispose([x, y, auxY]);
      return losses;
  
    }
  
    async trainSCombinedModelOneStep(batchSize) {
  
      const zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1); // <-- noise
      const sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');
  
      const trick = tf.ones([batchSize, 1]).mul(this.softOne);
  
      const losses = await this.combinedModel.trainOnBatch([zVectors], [trick, sampledLabels]);
      tf.dispose([zVectors, sampledLabels, trick]);
      return losses;
  
    }

    async trainACDiscriminatorOneStep(xTrain, yTrain, batchStart, batchSize) {
  
      const imageBatch = xTrain.slice(batchStart, batchSize);
      const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');
  
      let zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1);
      let sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');
  
      const generatedImages = this.generator.predict([zVectors, sampledLabels], {batchSize: batchSize});
  
      const x = tf.concat([imageBatch, generatedImages], 0);
  
      const y = tf.tidy(() => {
        return tf.concat([tf.ones([batchSize, 1]).mul(this.softOne), tf.zeros([batchSize, 1])]);
      });
  
      const auxY = tf.concat([labelBatch, sampledLabels], 0);
  
      const losses = await this.discriminator.trainOnBatch(x, [y, auxY]);
      tf.dispose([x, y, auxY]);
      return losses;
  
    }
  
    async trainACCombinedModelOneStep(batchSize) {
  
      const zVectors = tf.randomUniform([batchSize, this.latentSize], -1, 1); // <-- noise
      const sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');
  
      const trick = tf.ones([batchSize, 1]).mul(this.softOne);
  
      const losses = await this.combinedModel.trainOnBatch([zVectors, sampledLabels], [trick, sampledLabels]);
      tf.dispose([zVectors, sampledLabels, trick]);
      return losses;
  
    }
 
    async train() {
  
      // const numBatches = Math.ceil(this.xTrain.shape[0] / this.batchSize);
      const numBatches = 1;
      let dgl = 0, dal = 0, gl = 0;
      for(let batch = 0; batch < numBatches; batch++) {
  
        const actualBatchSize = (batch + 1) * this.batchSize >= this.xTrain.shape[0] ? (this.xTrain.shape[0] - batch * this.batchSize) : this.batchSize;
        const info = this.Info[0]
        if (info.type === "GAN" || info.type === "gan") {
          const dLoss = await this.trainDiscriminatorOneStep(this.xTrain, batch * this.batchSize, actualBatchSize);
          const gLoss = await this.trainCombinedModelOneStep(2 * actualBatchSize);
          dgl = dLoss.toFixed(6);
          gl = gLoss.toFixed(6);

          console.log(`batch ${batch + 1}/${numBatches}: dLoss = ${dLoss.toFixed(6)}, gLoss = ${gLoss.toFixed(6)}`);
        }
        else if (info.type === "Conditional GAN") {
          const dLoss = await this.trainCDiscriminatorOneStep(this.xTrain, this.yTrain, batch * this.batchSize, actualBatchSize);
          const gLoss = await this.trainCCombinedModelOneStep(2 * actualBatchSize);
          dgl = dLoss.toFixed(6);
          gl = gLoss.toFixed(6);

          console.log(`batch ${batch + 1}/${numBatches}: dLoss = ${dLoss.toFixed(6)}, gLoss = ${gLoss.toFixed(6)}`);
        }
        else if (info.type === "Semi-Supervised GAN") {
          const dLoss = await this.trainSDiscriminatorOneStep(this.xTrain, this.yTrain, batch * this.batchSize, actualBatchSize);
          const gLoss = await this.trainSCombinedModelOneStep(2 * actualBatchSize);
          dgl = dLoss[0].toFixed(6);
          dal = dLoss[1].toFixed(6);
          gl = gLoss[0].toFixed(6);

          console.log(`batch ${batch + 1}/${numBatches}: dGenLoss = ${dLoss[0].toFixed(6)}, dAuxLoss = ${dLoss[1].toFixed(6)}, gLoss = ${gLoss[0].toFixed(6)}`);
        }
        else {
          const dLoss = await this.trainACDiscriminatorOneStep(this.xTrain, this.yTrain, batch * this.batchSize, actualBatchSize);
          const gLoss = await this.trainACCombinedModelOneStep(2 * actualBatchSize);
          dgl = dLoss[0].toFixed(6);
          dal = dLoss[1].toFixed(6);
          gl = gLoss[0].toFixed(6);

          console.log(`batch ${batch + 1}/${numBatches}: dGenLoss = ${dLoss[0].toFixed(6)}, dAuxLoss = ${dLoss[1].toFixed(6)}, gLoss = ${gLoss[0].toFixed(6)}`);
        }
      }
      
      const dgen_div = document.getElementById('dgl');
      const daux_div = document.getElementById('dal');
      const gl_div = document.getElementById('gl');

      const dgen_new = document.createElement('div');
      const daux_new = document.createElement('div');
      const gl_new = document.createElement('div');

      dgen_new.textContent = dgl.toString();
      daux_new.textContent = dal.toString();
      gl_new.textContent = gl.toString();

      dgen_div.appendChild(dgen_new);
      daux_div.appendChild(daux_new);
      gl_div.appendChild(gl_new);
    }
  
    async download() {
      await this.generator.save("downloads://model");
    }
  
    async generate(canvas) {

      if (this.generator.inputs.length === 2) {

        const combinedFakes = tf.tidy(() => {
          
          let latentVectorLength = this.generator.inputs[0].shape[1];
          let numRepeats = this.numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
          
          let latentDims = latentVectorLength;
          let zs = new Array(latentDims).fill(0).map(_ => Math.random());
          let singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
          let latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].
    
          // Generate one fake image for each digit.
          let sampledLabels = tf.tensor2d(Array.from({ length: this.numClasses }, (_, index) => index), [this.numClasses, 1]);
          // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
          let generatedImages = this.generator.predict([latentVectors, sampledLabels]).add(1).div(2);
    
          // Concatenate the images horizontally into a single image.
          let row = tf.concat(tf.unstack(generatedImages), 1);
    
          for (let i=0; i<this.numClasses; i++){
            latentVectorLength = this.generator.inputs[0].shape[1];
            numRepeats = this.numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
            
            latentDims = latentVectorLength;
            zs = new Array(latentDims).fill(0).map(_ => Math.random());
            singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
            latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].
      
            // Generate one fake image for each digit.
            sampledLabels = tf.tensor2d(Array.from({ length: this.numClasses }, (_, index) => index), [this.numClasses, 1]);
            // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
            generatedImages = this.generator.predict([latentVectors, sampledLabels]).add(1).div(2);
            
            let row_1 = tf.concat(tf.unstack(generatedImages), 1);
            
            row = tf.concat([row, row_1], 0);
          }
    
          return row;
        });
    
        let uint8Clamped = await tf.browser.toPixels(combinedFakes, canvas);
        tf.dispose(combinedFakes);
    
        return uint8Clamped;
      }
    
      else {
    
        const combinedFakes = tf.tidy(() => {
          
          let latentVectorLength = this.generator.inputs[0].shape[1];
          let numRepeats = this.numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
          
          let latentDims = latentVectorLength;
          let zs = new Array(latentDims).fill(0).map(_ => Math.random());
          let singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
          let latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].
    
          // Generate one fake image for each digit.
          // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
          let generatedImages = this.generator.predict(latentVectors).add(1).div(2);
    
          // Concatenate the images horizontally into a single image.
          let row = tf.concat(tf.unstack(generatedImages), 1);
    
          for (let i=0; i<this.numClasses; i++){
            latentVectorLength = this.generator.inputs[0].shape[1];
            numRepeats = this.numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
            
            latentDims = latentVectorLength;
            zs = new Array(latentDims).fill(0).map(_ => Math.random());
            singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
            latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].
      
            // Generate one fake image for each digit.
            // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
            generatedImages = this.generator.predict(latentVectors).add(1).div(2);
            
            let row_1 = tf.concat(tf.unstack(generatedImages), 1);
            
            row = tf.concat([row, row_1], 0);
          }
    
          return row;
        });
    
        let uint8Clamped = await tf.browser.toPixels(combinedFakes, canvas);
        tf.dispose(combinedFakes);
    
        return uint8Clamped;
      }
  }
}

  async function loadMnistData() {
    // This code is based on code from here: https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
    const IMAGE_H = 28;
    const IMAGE_W = 28;
    const IMAGE_SIZE = IMAGE_H * IMAGE_W;
    const NUM_CLASSES = 10;
    const NUM_DATASET_ELEMENTS = 65000;
  
    const NUM_TRAIN_ELEMENTS = 55000;
    // const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
  
    const MNIST_IMAGES_SPRITE_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
    const MNIST_LABELS_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";
    
    let datasetImages;
    
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = "";
      img.onerror = reject;
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;
  
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
  
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;
  
        for(let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize);
          ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);
  
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
          for(let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        datasetImages = new Float32Array(datasetBytesBuffer);
  
        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
  
    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
  
    let datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
  
    // Slice the the images and labels into train and test sets.
    let trainImages = datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    let trainLabels = datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);

    // image data is in range [0, 1] and so we convert to [-1, 1] with .sub(0.5).mul(2)
    const images = tf.tensor4d(trainImages, [trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]).sub(0.5).mul(2);
    const labels = tf.tensor2d(trainLabels, [trainLabels.length / NUM_CLASSES, NUM_CLASSES]);

    // images: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
    // labels: The one-hot encoded labels tensor, of shape `[numTrainExamples, 10]`.
    return {images, labels};
  }

  async function loadFashionMnistData() {
    // This code is based on code from here: https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
    const IMAGE_H = 28;
    const IMAGE_W = 28;
    const IMAGE_SIZE = IMAGE_H * IMAGE_W;
    const NUM_CLASSES = 10;
    const NUM_DATASET_ELEMENTS = 65000;
  
    const NUM_TRAIN_ELEMENTS = 55000;
    // const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
  
    const MNIST_IMAGES_SPRITE_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png";
    const MNIST_LABELS_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8";
    
    let datasetImages;
    
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", {willReadFrequently: true});
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = "";
      img.onerror = reject;
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;
  
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
  
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;
  
        for(let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize);
          ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);
  
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
          for(let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        datasetImages = new Float32Array(datasetBytesBuffer);
  
        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
  
    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);
  
    let datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
  
    // Slice the the images and labels into train and test sets.
    let trainImages = datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    let trainLabels = datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);

    // image data is in range [0, 1] and so we convert to [-1, 1] with .sub(0.5).mul(2)
    const images = tf.tensor4d(trainImages, [trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]).sub(0.5).mul(2);
    const labels = tf.tensor2d(trainLabels, [trainLabels.length / NUM_CLASSES, NUM_CLASSES]);

    // images: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
    // labels: The one-hot encoded labels tensor, of shape `[numTrainExamples, 10]`.
    return {images, labels};
  }

  async function getData(i, l) {
    // const IMAGES_PATH = './data/train-images-idx3-ubyte';
    // const LABELS_PATH = './data/train-labels-idx1-ubyte';
    
    // const i_response = await fetch(i);
    // const i_buffer = await i_response.arrayBuffer();
    const i_buffer = await i.arrayBuffer();
    const data = new DataView(i_buffer);
    const numItems = data.getUint32(4, false);
    const numRows = data.getUint32(8, false);
    const numCols = data.getUint32(12, false);
    const iOffset = 16;

    // const l_response = await fetch(l);
    // const l_buffer = await l_response.arrayBuffer();
    const l_buffer = await l.arrayBuffer();
    const lOffset = 8;

    let imageData = new Uint8Array(i_buffer, iOffset);
    let datasetLabels = new Uint8Array(l_buffer, lOffset);
    const numClasses = new Set(datasetLabels);

    let trainImages = imageData.slice(0, numRows * numCols * numItems);
    let trainLabels = datasetLabels.slice(0, numClasses.size * numItems);
    
    const images = tf.tensor4d(trainImages, [numItems, numRows, numCols, 1]).sub(0.5).mul(2);
    const labels = tf.oneHot(tf.tensor2d(trainLabels, [trainLabels.length, 1]).flatten(), numClasses.size);
    
    return { images, labels };
  }