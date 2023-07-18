import Navbar from './Navbar';
import CardCarousel from './CardCarousel';
import CardCarouselSecond from './CardCarouselSecond';
import Grid from './Grid';
import Gen from './Gen';

import trainagan from './assets/train-a-gan.png'
import traingan from './assets/traingan.gif'
import './App.css';

function App() {
  return (
    <div className="App">
      {/* <Navbar />
      <div className="image-container">
        <img className="trainagan" src={trainagan} alt="train-a-gan" />
        <div class="overlay"></div>
        <img className="traingif" src={traingan} alt="train-gan-gif" />
      </div> */}

      {/* <CardCarousel />
      <CardCarouselSecond /> */}
      <Grid />
      {/* <Gen /> */}
    </div>
  );
}

export default App;
