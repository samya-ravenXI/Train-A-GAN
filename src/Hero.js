import React from 'react';

import trainAGan from './assets/train-a-gan.png';
import './Hero.css';

function Hero() {
  return (
    <div id="hero-section" className="image-container">        
      <img className="traingif" src={trainAGan} alt="train-gan-gif" />
    </div>
  )
}

export default Hero;