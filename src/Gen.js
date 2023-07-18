import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import './Gen.css';

function Gen() {

  const [formFields, setFormFields] = useState([
    { classes: 10 },
  ])

  const handleFormChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        alert('Total Classes should be between [1, 100]! Any value outside the boundary will be clipped between [1, 100].');
        newValue = 1;
    } else if (newValue > 100) {
        alert('Total Classes should be between [1, 100]! Any value outside the boundary will be clipped between [1, 100].');
        newValue = 100;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const submit = (e) => {
    e.preventDefault();
    console.log(formFields)
  }

  async function gen() {
    const generatorModel = await tf.loadLayersModel('./pretrained_model/model.json');
    generatorModel.summary();
    await generateAndDisplay(generatorModel);
  }

  async function generateAndDisplay(generatorModel) {
    let canvas = document.createElement("canvas");
    canvas.style.display = "block";
    await generate(generatorModel, canvas)
    const samples = document.getElementById("samples");
    samples.innerHTML = '';
    samples.appendChild(canvas);
  }

  async function generate(generator, canvas) {

    const numClasses = formFields[0].classes;

    if (generator.inputs.length === 2) {

    const combinedFakes = tf.tidy(() => {
      
      let latentVectorLength = generator.inputs[0].shape[1];
      let numRepeats = numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
      
      let latentDims = latentVectorLength;
      let zs = new Array(latentDims).fill(0).map(_ => Math.random());
      let singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
      let latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].

      // Generate one fake image for each digit.
      let sampledLabels = tf.tensor2d(Array.from({ length: numClasses }, (_, index) => index), [numClasses, 1]);
      // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
      let generatedImages = generator.predict([latentVectors, sampledLabels]).add(1).div(2);

      // Concatenate the images horizontally into a single image.
      let row = tf.concat(tf.unstack(generatedImages), 1);

      for (let i=0; i<numClasses; i++){
        latentVectorLength = generator.inputs[0].shape[1];
        numRepeats = numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
        
        latentDims = latentVectorLength;
        zs = new Array(latentDims).fill(0).map(_ => Math.random());
        singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
        latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].
  
        // Generate one fake image for each digit.
        sampledLabels = tf.tensor2d(Array.from({ length: numClasses }, (_, index) => index), [numClasses, 1]);
        // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
        generatedImages = generator.predict([latentVectors, sampledLabels]).add(1).div(2);
        
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
      
      let latentVectorLength = generator.inputs[0].shape[1];
      let numRepeats = numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
      
      let latentDims = latentVectorLength;
      let zs = new Array(latentDims).fill(0).map(_ => Math.random());
      let singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
      let latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].

      // Generate one fake image for each digit.
      // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
      let generatedImages = generator.predict(latentVectors).add(1).div(2);

      // Concatenate the images horizontally into a single image.
      let row = tf.concat(tf.unstack(generatedImages), 1);

      for (let i=0; i<numClasses; i++){
        latentVectorLength = generator.inputs[0].shape[1];
        numRepeats = numClasses; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
        
        latentDims = latentVectorLength;
        zs = new Array(latentDims).fill(0).map(_ => Math.random());
        singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
        latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].
  
        // Generate one fake image for each digit.
        // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval [0, 1].
        generatedImages = generator.predict(latentVectors).add(1).div(2);
        
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

  return (
    <div className="Gen">
      <div id='input'>
        <form onSubmit={submit}>
          {formFields.map((form, index) => {
            return (
              <div className='info' key={index}>
                  <div>
                    <label>
                      <p>Total Classes</p>
                    <input
                        className = 'c'
                        type='number'
                        min='1'
                        max='100'
                        name='classes'
                        placeholder='Classes'
                        onChange={event => handleFormChange(event, index)}
                        value={form.classes}
                    />
                    </label>
                  </div>
              </div>
            )
          })}
        </form>
        <button id='genBtn' className='sub' onClick={gen}>Generate Samples</button>
      </div>
      <div id='samples'></div>
    </div>
  );
}

export default Gen;