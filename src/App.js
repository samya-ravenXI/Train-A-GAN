import Navbar from './Navbar';
import CardCarousel from './CardCarousel';
import CardCarouselSecond from './CardCarouselSecond';
import Grid from './Grid';
import Gen from './Gen';

import './App.css';
import Hero from './Hero';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Hero />
      <CardCarousel />
      <CardCarouselSecond />
      {/* <Grid />
      <Gen /> */}
    </div>
  );
}

export default App;
