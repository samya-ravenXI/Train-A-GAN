import React from 'react';

import combined_hero from './assets/combined_hero.gif';
import './Hero.css';

function Hero() {
  return (
    <div id="hero-section" className="image-container">        
      <img className="traingif" src={combined_hero} alt="train-gan-gif" />
    </div>
  )
}

export default Hero;