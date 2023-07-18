import React, { useState } from 'react';
import './Navbar.css';
import logo from './assets/logo.png';

function Navbar() {

    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const handleMenuToggle = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    return (
        <nav className={`navbar ${isMenuOpen ? 'menu-open' : ''}`}>
            <div className="navbar-menu disable-select" onClick={handleMenuToggle}>
                <div className={`navbar-logo ${isMenuOpen ? 'inverted' : ''}`}>
                    <img src={logo} alt="logo" />
                </div>
            </div>

            {isMenuOpen && (
                <div className="sidebar">
                    <a href="#hero-section">Section 1</a>
                    <a href="#">Section 2</a>
                    <a href="#">Section 3</a>
                    <a href="#">Section 4</a>
                    <a href="#">Section 5</a>
                </div>
            )}

            <ul className="navbar-menu">
                <div class="dropdown">
                    <button class="dropbtn">Resources
                        <i class="fa fa-caret-down"></i>
                    </button>
                    <div class="dropdown-content">
                        <a href="#" target="_blank">GAN</a>
                        <a href="#" target="_blank">Conditional GAN</a>
                        <a href="#" target="_blank">Semi-supervised GAN</a>
                        <a href="#" target="_blank">Auxiliary Classifier GAN</a>
                    </div>
                </div>
                
                <div class="dropdown">
                    <button class="dropbtn">Source
                        <i class="fa fa-caret-down"></i>
                    </button>
                    <div class="dropdown-content">
                        <a href="#" target="_blank">GAN</a>
                        <a href="#" target="_blank">Conditional GAN</a>
                        <a href="#" target="_blank">Semi-supervised GAN</a>
                        <a href="#" target="_blank">Auxiliary Classifier GAN</a>
                        <a href="#" target="_blank">tf.js Training Example</a>
                        <a href="#" target="_blank">ACGAN MNIST TensorFlow Example</a>
                    </div>
                </div>
                
                <a>About</a>
            </ul>
        </nav>
    ); 
}

export default Navbar;
