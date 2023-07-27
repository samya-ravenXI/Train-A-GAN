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
        // alert('Total Classes should be between [1, 100]!');
        showCustomAlert("Total Classes should be between [1, 100]!");
        newValue = 1;
    } else if (newValue > 100) {
        // alert('Total Classes should be between [1, 100]!');
        showCustomAlert("Total Classes should be between [1, 100]!");
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

  // async function gen() {
  //   const generatorModel = await tf.loadLayersModel('./pretrained_model/model.json');
  //   await generateAndDisplay(generatorModel);
  // }

  async function generateAndDisplay(generatorModel) {
    let canvas = document.createElement("canvas");
    // canvas.style.display = "block";
    await generate(generatorModel, canvas)
    const samples = document.getElementById("gen-samples");
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

async function loadModel() {
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');
  try {
    const model = await tf.loadLayersModel(tf.io.browserFiles(
      [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
    await generateAndDisplay(model);
  }
  catch (error){
    showCustomAlert("Error! Please check whether the correct architecture and its weights have been provided. Check console for details.");
    console.log(error);
  }
}

function disableInputs() {
  // Disable all input, select, and button elements
  const inputs = document.querySelectorAll("input, select, button:not(#gen-alert-button)");
  inputs.forEach((input) => {
    input.disabled = true;
  });
}

function enableInputs() {
  // Enable all input, select, and button elements
  const inputs = document.querySelectorAll("input, select, button");
  inputs.forEach((input) => {
    input.disabled = false;
  });
}

function showCustomAlert(message) {
  const customAlert = document.getElementById("gen-alert");
  const alertMessage = document.getElementById("gen-alert-msg");

  alertMessage.textContent = message;
  customAlert.style.display = "block";
  disableInputs(); 
}

function hideCustomAlert() {
  const customAlert = document.getElementById("gen-alert");
  customAlert.style.display = "none";
  enableInputs();
}

  return (
    <div className="Gen">
      <div className="custom-alert" id="gen-alert">
        <div className="gen-alert-content">
          <p id="gen-alert-msg"></p>
          <button id="gen-alert-button" onClick={hideCustomAlert}>Confirm</button>
        </div>
      </div>
      <h1>Generate</h1>
      <div className="gen-container">
        <div id='input'>
          <form onSubmit={submit}>
            {formFields.map((form, index) => {
              return (
                <div className='gen-info' key={index}>
                    <h3>Total Classes</h3>
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
                </div>
              );
            })}
          </form>
          <div className='files'>
            <h3>Model Architecture</h3>
            <h3>Model Weights</h3>
            <input type="file" id="upload-json"/>
            <input type="file" id="upload-weights"/>
          </div>
          <button id='genBtn' className='sub' onClick={loadModel}>Generate Samples</button>
        </div>
        <div id='sample-container'>
          <div></div>
          <div id='gen-samples'></div>
          <div></div>
        </div>
      </div>
    </div>
  );
}

export default Gen;