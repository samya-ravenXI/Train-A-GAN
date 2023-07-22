import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Script from './script';
import GAN from './model';
import './Grid.css';

function Grid() {

  const [formFields, setFormFields] = useState([
    { data: 'mnist', type: 'gan', init: 'glorotNormal', ishape_h: 28, ishape_w: 28, ishape_d:1, reshape_h: 7, reshape_w: 7, reshape_d: 512, epoch: 5, images: '', labels: ''},
  ])
  const [formGenFields, setFormGenFields] = useState([
    { depth: 64, kernel_size: 3, stride: 1, padding: 'same', batch_normalisation: 'no', activation: 'relu', dropout: 0.2},
  ])
  const [formDisFields, setFormDisFields] = useState([
    { depth: 64, kernel_size: 3, stride: 1, padding: 'same', batch_normalisation: 'no', activation: 'relu', dropout: 0.2},
  ])

  const handleFormChange = (event, index) => {
    const value = event.target.value;
    const upload = document.getElementById("upload");
    if (value === 'Custom'){
      showUpload();

    }
    else if (value === 'Mnist' || value === 'Fashion Mnist' || value === 'mist' ) {
      hideUpload();
    }
    let data = [...formFields];
    data[index][event.target.name] = event.target.value;
    setFormFields(data);
  }

  const handleImagesFormChange = (event, index) => {

    let data = [...formFields];
    if (data[0].data === 'Custom'){
      data[index][event.target.name] = event.target.files[0];
    }
    else if (data[0].data === 'Mnist' | data[0].data === 'Fashion Mnist' | data[0].data === 'mist' ) {
      data[index][event.target.name] = '';
    }
    data[index][event.target.name] = event.target.files[0];
    setFormFields(data);
  }
  
  const handleLabelsFormChange = (event, index) => {

    const value = event.target.value;
    let data = [...formFields];
    if (data[0].data === 'Custom'){
      data[index][event.target.name] = event.target.files[0];
    }
    else if (data[0].data === 'Mnist' | data[0].data === 'Fashion Mnist' | data[0].data === 'mist' ) {
      data[index][event.target.name] = '';
    }
    data[index][event.target.name] = event.target.files[0];
    setFormFields(data);
  }

  const handleFormEpochChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Epoch should be between [1, 100]!');
        newValue = 1;
    } else if (newValue > 100) {
        showCustomAlert('Epoch should be between [1, 100]!');
        newValue = 100;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormInputShapeHChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Input Shape Height should be between [1, 128]!');
        newValue = 1;
    } else if (newValue > 128) {
        showCustomAlert('Input Shape Height should be between [1, 128]!');
        newValue = 128;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormInputShapeWChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Input Shape Width should be between [1, 128]!');
        newValue = 1;
    } else if (newValue > 128) {
        showCustomAlert('Input Shape Width should be between [1, 128]!');
        newValue = 128;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormInputShapeDChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Input Shape Depth should be between [1, 15]!');
        newValue = 1;
    } else if (newValue > 15) {
        showCustomAlert('Input Shape Depth should be between [1, 15]!');
        newValue = 15;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormReshapeHChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Reshape Height should be between [1, 64]!');
        newValue = 1;
    } else if (newValue > 64) {
        showCustomAlert('Reshape Height should be between [1, 64]!');
        newValue = 64;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormReshapeWChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Reshape Width should be between [1, 64]!');
        newValue = 1;
    } else if (newValue > 64) {
        showCustomAlert('Reshape Width should be between [1, 64]!');
        newValue = 64;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormReshapeDChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Reshape Depth should be between [1, 512]!');
        newValue = 1;
    } else if (newValue > 512) {
        showCustomAlert('Reshape Depth should be between [1, 512]!');
        newValue = 512;
    }
    let data = [...formFields];
    data[index][event.target.name] = newValue;
    setFormFields(data);
  }

  const handleFormGenChange = (event, index) => {
    let data = [...formGenFields];
    data[index][event.target.name] = event.target.value;
    setFormGenFields(data);
  }

  const handleFormDepthGenChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Depth should be between [1, 512]!');
        newValue = 1;
    } else if (newValue > 512) {
        showCustomAlert('Depth should be between [1, 512]!');
        newValue = 512;
    }
    let data = [...formGenFields];
    data[index][event.target.name] = newValue;
    setFormGenFields(data);
  }

  const handleFormKernelSizeGenChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Kernel Size should be between [1, 15]!');
        newValue = 1;
    } else if (newValue > 15) {
        showCustomAlert('Kernel Size should be between [1, 15]!');
        newValue = 15;
    }
    let data = [...formGenFields];
    data[index][event.target.name] = newValue;
    setFormGenFields(data);
  }

  const handleFormStrideGenChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Stride should be between [1, 10]!');
        newValue = 1;
    } else if (newValue > 10) {
        showCustomAlert('Stride should be between [1, 10]!');
        newValue = 10;
    }
    let data = [...formGenFields];
    data[index][event.target.name] = newValue;
    setFormGenFields(data);
  }

  const handleFormDropoutGenChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = parseFloat(value);
    if (newValue < 0) {
        showCustomAlert('Dropout should be between [0, 1]!');
        newValue = 0;
    } else if (newValue > 1) {
        showCustomAlert('Dropout should be between [0, 1]!');
        newValue = 1;
    }
    let data = [...formGenFields];
    data[index][event.target.name] = newValue;
    setFormGenFields(data);
  }


  const handleFormDisChange = (event, index) => {
    let data = [...formDisFields];
    data[index][event.target.name] = event.target.value;
    setFormDisFields(data);
  }

  const handleFormDepthDisChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Depth should be between [1, 512]!');
        newValue = 1;
    } else if (newValue > 512) {
        showCustomAlert('Depth should be between [1, 512]!');
        newValue = 512;
    }
    let data = [...formDisFields];
    data[index][event.target.name] = newValue;
    setFormDisFields(data);
  }

  const handleFormKernelSizeDisChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Kernel Size should be between [1, 15]!');
        newValue = 1;
    } else if (newValue > 15) {
        showCustomAlert('Kernel Size should be between [1, 15]!');
        newValue = 15;
    }
    let data = [...formDisFields];
    data[index][event.target.name] = newValue;
    setFormDisFields(data);
  }

  const handleFormStrideDisChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = Number(value);
    if (newValue < 1) {
        showCustomAlert('Stride should be between [1, 10]!');
        newValue = 1;
    } else if (newValue > 10) {
        showCustomAlert('Stride should be between [1, 10]!');
        newValue = 10;
    }
    let data = [...formDisFields];
    data[index][event.target.name] = newValue;
    setFormDisFields(data);
  }

  const handleFormDropoutDisChange = (event, index) => {
    
    const value = event.target.value;
    let newValue = parseFloat(value);
    if (newValue < 0) {
        showCustomAlert('Dropout should be between [0, 1]!');
        newValue = 0;
    } else if (newValue > 1) {
        showCustomAlert('Dropout should be between [0, 1]!');
        newValue = 1;
    }
    let data = [...formDisFields];
    data[index][event.target.name] = newValue;
    setFormDisFields(data);
  }

  const submit = (e) => {
    e.preventDefault();
    console.log(formFields)
    console.log(formGenFields)
    console.log(formDisFields)
  }

  const addGenFields = () => {
    let object = {
      depth: 64,
      kernel_size: 3,
      stride: 1,
      padding: 'same', 
      batch_normalisation: 'no', 
      activation: 'relu',
      dropout: 0.2
    }

    setFormGenFields([...formGenFields, object])
  }

  const addDisFields = () => {
    let object = {
      depth: 64,
      kernel_size: 3,
      stride: 1,
      padding: 'same', 
      batch_normalisation: 'no', 
      activation: 'relu',
      dropout: 0.2
    }

    setFormDisFields([...formDisFields, object])
  }

  const removeGenFields = (index) => {
    let data = [...formGenFields];
    data.splice(index, 1)
    setFormGenFields(data)
  }

  const removeDisFields = (index) => {
    let data = [...formDisFields];
    data.splice(index, 1)
    setFormDisFields(data)
  }
  
  function showUpload(message) {
    const upload = document.getElementById("upload");
    upload.style.display = "block";
    disableInputs(); 
  }
  
  function hideUpload() {
    const upload = document.getElementById("upload");
    upload.style.display = "none";
    enableInputs();
  }

  let gan;
  let trainingDone = false;
  let startedTraining = false;
  const ep = document.getElementById('ep');
  const dgen_div = document.getElementById('dgl');
  const daux_div = document.getElementById('dal');
  const gl_div = document.getElementById('gl');

  async function train() {
    try {
      ep.innerHTML = '';
      dgen_div.innerHTML = '';
      daux_div.innerHTML = '';
      gl_div.innerHTML = '';
    }
    catch(e) {
      
    }

    const epochCount = formFields[0].epoch;
    gan = new GAN({Gen: formGenFields, Dis: formDisFields, Info: formFields});
    startedTraining = true;
    let currEpoch = 1;
    console.log(formFields);
    if(formFields[0].data === 'Custom' && (formFields[0].images === '' || formFields[0].labels === '')) {
      showCustomAlert('Please upload the data files first!');
    }
    else{

      await tf.setBackend("webgl");
      await gan.setTrainingData(formFields[0].data, formFields[0].images, formFields[0].labels);

      // while (currEpoch <= epochCount) {
      //   try {
      //     await gan.train();
      //     await generate();
      //     console.log(`====== Finished Epoch ${currEpoch} ======`);
      //     const ep_new = document.createElement('div');
      //     ep_new.textContent = currEpoch.toString();
      //     ep.appendChild(ep_new);
      //     currEpoch++;
      //   } catch (error) {
      //     showCustomAlert('Please recheck the entered values. Check console for details.');
      //     console.log(error);
      //     break;
      //   }
      
      // }
      trainingDone = true;
    }
  }

  async function generate() {
    const genSamples = document.getElementById("samples");
    if(!startedTraining) {
      return alert("You need to start training first. Click the 'train model' button.");
    }

    let canvas = document.createElement("canvas");
    canvas.id = 'sample';
    await gan.generate(canvas);
    genSamples.innerHTML = '';
    genSamples.appendChild(canvas);
  }

  async function download() {
    if(!trainingDone) {
      return showCustomAlert("You need to start training first! Click the 'Start Training' button.");
    }
    await gan.download();
  }

  let sc;
  async function script() {
    sc = new Script({Gen: formGenFields, Dis: formDisFields, Info: formFields});
    sc.downloadTextFile();
  }

  function disableInputs() {
    // Disable all input, select, and button elements
    const inputs = document.querySelectorAll("input:not(#upload-images, #upload-labels), select, button:not(#grid-alert-button, #upload-button)");
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
    const customAlert = document.getElementById("grid-alert");
    const alertMessage = document.getElementById("grid-alert-msg");
  
    alertMessage.textContent = message;
    customAlert.style.display = "block";
    disableInputs(); 
  }
  
  function hideCustomAlert() {
    const customAlert = document.getElementById("grid-alert");
    customAlert.style.display = "none";
    enableInputs();
  }

  return (
    <div className="Grid">
      <div className="custom-alert" id="grid-alert">
        <div className="grid-alert-content">
          <p id="grid-alert-msg"></p>
          <button id="grid-alert-button" onClick={hideCustomAlert}>Confirm</button>
        </div>
      </div>
      <form onSubmit={submit}>
        {formFields.map((form, index) => {
          return (
            <div id="upload" key={index}>
              <div id="upload-mat">
                <div id="upload-files">
                  <h3>Images</h3>
                  <h3>Labels</h3>
                  <input name="images" type="file" id="upload-images" onChange={event => handleImagesFormChange(event, index)}/>
                  <input name="labels" type="file" id="upload-labels" onChange={event => handleLabelsFormChange(event, index)}/>
                </div>
                <button id="upload-button" onClick={hideUpload}>Confirm</button>
              </div>
            </div>
          )
        })}
      </form>
      <h1>Train</h1>
      <form onSubmit={submit}>
        {formFields.map((form, index) => {
          return (
            <div className='info' key={index}>
                <h3>Dataset</h3>
                <h3>GAN Type</h3>
                <select
                    aria-label="h3 for the select"
                    name='data'
                    placeholder='Dataset'
                    onChange={event => handleFormChange(event, index)}
                    value={form.data}
                >
                    <option>Mnist</option>
                    <option>Custom</option>
                    <option>Fashion Mnist</option>
                </select>
                <select
                    aria-label="h3 for the select"
                    name='type'
                    placeholder='Type'
                    onChange={event => handleFormChange(event, index)}
                    value={form.type}
                >
                    <option>GAN</option>
                    <option>Conditional GAN</option>
                    <option>Semi-Supervised GAN</option>
                    <option>Auxiliary Classifier GAN</option>
                </select>
                <div>
                <div className='inf'>
                  <h3>Epoch</h3>
                  <input
                      id = 'e'
                      type='number'
                      min='1'
                      max='100'
                      name='epoch'
                      placeholder='Epoch'
                      onChange={event => handleFormEpochChange(event, index)}
                      value={form.epoch}
                  />
                </div>
                <div className='inf'>
                  <h3>Initiliazer</h3>
                  <select
                    id = 'init'
                    aria-label="h3 for the select"
                    name='init'
                    placeholder='Init'
                    onChange={event => handleFormChange(event, index)}
                    value={form.init}
                >
                    <option>heNormal</option>
                    <option>randomNormal</option>
                    <option>glorotNormal</option>
                  </select>
                </div>
                </div>
                <div className='shape'>
                    <div>
                      <h3>Input Shape</h3>
                      <div className='ishape'>
                          <input
                            type='number'
                            min='1'
                            max='64'
                            name='ishape_h'
                            placeholder='H'
                            onChange={event => handleFormInputShapeHChange(event, index)}
                            value={form.ishape_h}
                          />
                          <input
                            type='number'
                            min='1'
                            max='64'
                            name='ishape_w'
                            placeholder='W'
                            onChange={event => handleFormInputShapeWChange(event, index)}
                            value={form.ishape_w}
                          />
                          <input
                            type='number'
                            min='1'
                            max='15'
                            name='ishape_d'
                            placeholder='D'
                            onChange={event => handleFormInputShapeDChange(event, index)}
                            value={form.ishape_d}
                          />
                      </div>
                    </div>
                    <div>
                      <h3>Reshape Dense Shape</h3>
                      <div className='reshape'>
                          <input
                            type='number'
                            min='1'
                            max='64'
                            name='reshape_h'
                            placeholder='H'
                            onChange={event => handleFormReshapeHChange(event, index)}
                            value={form.reshape_h}
                          />
                          <input
                            type='number'
                            min='1'
                            max='64'
                            name='reshape_w'
                            placeholder='W'
                            onChange={event => handleFormReshapeWChange(event, index)}
                            value={form.reshape_w}
                          />
                          <input
                            type='number'
                            min='1'
                            max='512'
                            name='reshape_d'
                            placeholder='D'
                            onChange={event => handleFormReshapeDChange(event, index)}
                            value={form.reshape_d}
                          />
                      </div>
                    </div>
                </div>
            </div>
          )
        })}
      </form>
      <div className="gen">
      <div className="head header">
        <h3>Depth</h3>
        <h3>Kernel</h3>
        <h3>Stride</h3>
        <h3>Padding</h3>
        <h3>BatchNorm</h3>
        <h3>Activation</h3>
        <h3>Dropout</h3>
      </div>
        <form onSubmit={submit}>
        {formGenFields.map((form, index) => {
          return (
            <div className="head" key={index}>
              <input
                className = 'd'
                type='number'
                min='1'
                max='512'
                name='depth'
                placeholder='Depth'
                onChange={event => handleFormDepthGenChange(event, index)}
                value={form.depth}
              />
              <input
                name='kernel_size'
                placeholder='Kernel Size'
                onChange={event => handleFormKernelSizeGenChange(event, index)}
                value={form.kernel_size}
              />
              <input
                name='stride'
                placeholder='Stride'
                onChange={event => handleFormStrideGenChange(event, index)}
                value={form.stride}
              />
              <select
                aria-label="h3 for the select"
                name='padding'
                placeholder='Padding'
                onChange={event => handleFormGenChange(event, index)}
                value={form.padding}
              >
                <option>Same</option>
                <option>Valid</option>
              </select>
              <select
                aria-label="h3 for the select"
                name='batch_normalisation'
                placeholder='Batch Normalisation'
                onChange={event => handleFormGenChange(event, index)}
                value={form.batch_normalisation}
              >
                <option>No</option>
                <option>Yes</option>
              </select>
              <select
                aria-label="h3 for the select"
                name='activation'
                placeholder='Activation'
                onChange={event => handleFormGenChange(event, index)}
                value={form.activation}
              >
                <option>ReLU</option>
                <option>Tanh</option>
                <option>LeakyReLU</option>
              </select>
              <input
                type='number'
                name='dropout'
                placeholder='Dropout'
                onChange={event => handleFormDropoutGenChange(event, index)}
                value={form.dropout}
              />
              <button className='min' onClick={() => removeGenFields(index)}>-</button>
            </div>
          )
        })}
        </form>
        <button className='add' onClick={addGenFields}>+</button>
      </div>
      <div className="dis">
        <form onSubmit={submit}>
            {formDisFields.map((form, index) => {
            return (
                <div className="head" key={index}>
                <input
                    className = 'd'
                    type='number'
                    min='1'
                    max='512'
                    name='depth'
                    placeholder='Depth'
                    onChange={event => handleFormDepthDisChange(event, index)}
                    value={form.depth}
                />
                <input
                    name='kernel_size'
                    placeholder='Kernel Size'
                    onChange={event => handleFormKernelSizeDisChange(event, index)}
                    value={form.kernel_size}
                />
                <input
                    name='stride'
                    placeholder='Stride'
                    onChange={event => handleFormStrideDisChange(event, index)}
                    value={form.stride}
                />
                <select
                    aria-label="h3 for the select"
                    name='padding'
                    placeholder='Padding'
                    onChange={event => handleFormDisChange(event, index)}
                    value={form.padding}
                >
                    <option>Same</option>
                    <option>Valid</option>
                </select>
                <select
                    aria-label="h3 for the select"
                    name='batch_normalisation'
                    placeholder='Batch Normalisation'
                    onChange={event => handleFormDisChange(event, index)}
                    value={form.batch_normalisation}
                >
                    <option>No</option>
                    <option>Yes</option>
                </select>
                <select
                    aria-label="h3 for the select"
                    name='activation'
                    placeholder='Activation'
                    onChange={event => handleFormDisChange(event, index)}
                    value={form.activation}
                >
                    <option>ReLU</option>
                    <option>Tanh</option>
                    <option>LeakyReLU</option>
                </select>
                <input
                    type='number'
                    name='dropout'
                    placeholder='Dropout'
                    onChange={event => handleFormDropoutDisChange(event, index)}
                    value={form.dropout}
                />
                <button className='min' onClick={() => removeDisFields(index)}>-</button>
                </div>
            )
            })}
        </form>
        <button className='add' onClick={addDisFields}>+</button>
      </div>
      <div className='buttons'>
        <button id='scriptBtn' className='sub' onClick={script}>Download Script</button>
        <button id='trainBtn' className='sub' onClick={train}>Start Training</button>
        <button id='downBtn' className='sub' onClick={download}>Download Model</button>
      </div>
      <div className='line'></div>
      <h1>History</h1>
      <div id='output'>
        <div id='losses'>
          <h3>Epoch</h3>
          <h3>Discriminator Generation Loss</h3>
          <h3>Discriminator Auxiliary Loss</h3>
          <h3>Generator Generation Loss</h3>
          <div id="ep"></div>
          <div id="dgl"></div>
          <div id="dal"></div>
          <div id="gl"></div>
        </div>
        <div id='samples'></div>
      </div>
    </div>
  );
}

export default Grid;