import * as tf from '@tensorflow/tfjs';

const TRAIN_IMAGES_FILE = './data/train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = './data/train-labels-idx1-ubyte';
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

function fetchOnceAndSaveToDiskWithBuffer(filename) {
    return new Promise((resolve, reject) => {
      fetch(filename)
        .then(response => response.arrayBuffer())
        .then(buffer => {
          resolve(new Uint8Array(buffer));
        })
        .catch(error => reject(error));
    });
  }
  
async function loadImages(filename) {
    const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);
    const headerBytes = IMAGE_HEADER_BYTES;
    const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;
  
    const images = [];
    let index = headerBytes;
    while (index < buffer.length) {
      const array = new Float32Array(recordBytes);
      for (let i = 0; i < recordBytes; i++) {
        // Normalize the pixel values into the 0-1 interval, from
        // the original [-1, 1] interval.
        array[i] = (buffer[index++] - 127.5) / 127.5;
      }
      images.push(array);
    }
  
    return images;
  }

  async function loadLabels(filename) {
    const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);
  
    const headerBytes = LABEL_HEADER_BYTES;
    const recordBytes = LABEL_RECORD_BYTE;
  
    const labels = [];
    let index = headerBytes;
    while (index < buffer.length) {
      const array = new Int32Array(recordBytes);
      for (let i = 0; i < recordBytes; i++) {
        array[i] = buffer[index++];
      }
      labels.push(array);
    }
    return labels;
  }
  

export default class Dataset {

    constructor(opts={}) {
      
        this.dataset = null;
    }
  
    // opts is "MNIST" or {images, labels} where:
    //   images: tensor with shape [numImages, width, height, 1], but width must equal height (images must be square)
    //   labels: tensor with shape [numImages, numClasses] (i.e. an array of one-hot-encoded classes)
    async loadData(opts) {
        this.dataset = await Promise.all([
            loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE)
        ]);
    }

    getTrainData() {
        return this.getData_(true);
    }

    getData_(isTrainingData) {
        let imagesIndex;
        let labelsIndex;
        if (isTrainingData) {
          imagesIndex = 0;
          labelsIndex = 1;
        } else {
          imagesIndex = 2;
          labelsIndex = 3;
        }
        const size = this.dataset[imagesIndex].length;
        // Only create one big array to hold batch of images.
        const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
        const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
        const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));
    
        let imageOffset = 0;
        let labelOffset = 0;
        for (let i = 0; i < size; ++i) {
          images.set(this.dataset[imagesIndex][i], imageOffset);
          labels.set(this.dataset[labelsIndex][i], labelOffset);
          imageOffset += IMAGE_FLAT_SIZE;
          labelOffset += 1;
        }
    
        return {
          images: tf.tensor4d(images, imagesShape),
          labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
        };
    }
}