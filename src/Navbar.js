import React, { useState } from 'react';
import './Navbar.css';
import logo from './assets/logo.png';

function Navbar() {

    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const handleMenuToggle = () => {
    setIsMenuOpen(!isMenuOpen);
    };

    const [activePanel, setActivePanel] = useState('');

    const handlePanelHover = (panelName) => {
        setActivePanel(panelName);
    };

    return (
        <nav className={`navbar ${isMenuOpen ? 'menu-open' : ''}`}>
            <div className="navbar-menu" onClick={handleMenuToggle}>
                <div className={`navbar-logo ${isMenuOpen ? 'inverted' : ''}`}>
                    <img src={logo} alt="logo" />
                </div>
            </div>
            {isMenuOpen && (
            <div className="sidebar">
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
            </div>
            )}\
            <ul className="navbar-menu">
                <li
                className={`navbar-item ${activePanel === 'panel1' ? 'active' : ''}`}
                onMouseEnter={() => handlePanelHover('panel1')}
                onMouseLeave={() => handlePanelHover('')}
                >
                <a href=".">Resources</a>
                {activePanel === 'panel1' && (
                    <div className="panel">
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                    </div>
                )}
                </li>
                <li
                className={`navbar-item ${activePanel === 'panel2' ? 'active' : ''}`}
                onMouseEnter={() => handlePanelHover('panel2')}
                onMouseLeave={() => handlePanelHover('')}
                >
                <a href=".">Source</a>
                {activePanel === 'panel2' && (
                    <div className="panel">
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                    </div>
                )}
                </li>
                <li
                className={`navbar-item ${activePanel === 'panel3' ? 'active' : ''}`}
                onMouseEnter={() => handlePanelHover('panel3')}
                onMouseLeave={() => handlePanelHover('')}
                >
                <a href=".">About</a>
                {activePanel === 'panel3' && (
                    <div className="panel">
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                    </div>
                )}
                </li>
            </ul>
        </nav>
    ); 
}

export default Navbar;
