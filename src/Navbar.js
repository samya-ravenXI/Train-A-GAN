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
                <div className="dropdown">
                    <button className="dropbtn">Resources</button>
                    <div className="dropdown-content">
                        <a href="#" target="_blank">GAN</a>
                        <a href="#" target="_blank">Conditional GAN</a>
                        <a href="#" target="_blank">Semi-supervised GAN</a>
                        <a href="#" target="_blank">Auxiliary ClassNameifier GAN</a>
                    </div>
                </div>
                
                <div className="dropdown">
                    <button className="dropbtn">Source</button>
                    <div className="dropdown-content">
                        <a href="#" target="_blank">GAN</a>
                        <a href="#" target="_blank">Conditional GAN</a>
                        <a href="#" target="_blank">Semi-supervised GAN</a>
                        <a href="#" target="_blank">Auxiliary Classifier GAN</a>
                        <a href="#" target="_blank">tf.js Training Example</a>
                        <a href="#" target="_blank">ACGAN MNIST TensorFlow Example</a>
                    </div>
                </div>

                <div className="dropdown">
                    <button className="dropbtn">About</button>
                    <div className="github-card">
                        {/* TODO: After making the repo public make this part dynamic using GitHub API */}
                        <div className='header'>
                            <a target="_blank" href="https://github.com/samya-ravenXI/Train-A-GAN">Train-A-GAN</a>
                            <div className='pill'>Public</div>
                        </div>

                        <p className="description">Duis aliqua consequat ad eu officia dolore laborum laborum nulla eu ut laborum commodo incididunt nisi amet ut sed id incididunt ex veniam nisi dolor nostrud anim occaecat irure anim nisi aliquip voluptate fugiat enim aliquip tempor esse ullamco ad dolore esse laboris velit eu reprehenderit dolore id elit sint</p>
                    </div>
                </div>
            </ul>
        </nav>
    ); 
}

export default Navbar;
