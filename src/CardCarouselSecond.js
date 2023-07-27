import React, { useEffect, useCallback, useState } from "react";
import "./CardCarouselSecond.css";
import type1 from './assets/Type1.png'
import type2 from './assets/Type2.png'
import type3 from './assets/Type3.png'
import type4 from './assets/Type4.png'

const cardItems = [
  {
    id: 1,
    title: "Conditional GAN",
    aurthor: "Mirza & Osindero, 2014",
    image: type1,
    copy: "ut dolor non eiusmod est culpa quis voluptate quis magna officia eu in tempor non fugiat tempor irure cupidatat velit qui adipiscing pariatur cupidatat ut fugiat qui ullamco est sunt sit irure occaecat deserunt voluptate reprehenderit minim labore ullamco ullamco minim dolore fugiat ullamco magna quis veniam anim eiusmod laboris Duis culpa fugiat Excepteur labore laborum mollit qui ea consequat tempor tempor ut aute amet cupidatat ea occaecat sunt exercitation tempor cupidatat Duis aliquip ea cupidatat velit eiusmod id id dolore fugiat eiusmod id Excepteur incididunt id labore dolore eiusmod"
  },
  {
    id: 2,
    title: "Semi-Supervised GAN",
    aurthor: "Odina, 2016; Salimans, et al., 2016",
    image: type2,
    copy: "ut dolor non eiusmod est culpa quis voluptate quis magna officia eu in tempor non fugiat tempor irure cupidatat velit qui adipiscing pariatur cupidatat ut fugiat qui ullamco est sunt sit irure occaecat deserunt voluptate reprehenderit minim labore ullamco ullamco minim dolore fugiat ullamco magna quis veniam anim eiusmod laboris Duis culpa fugiat Excepteur labore laborum mollit qui ea consequat tempor tempor ut aute amet cupidatat ea occaecat sunt exercitation tempor cupidatat Duis aliquip ea cupidatat velit eiusmod id id dolore fugiat eiusmod id Excepteur incididunt id labore dolore eiusmod"
  },
  {
    id: 3,
    title: "InfoGAN",
    aurthor: "Chen et al., 2016",
    image: type3,
    copy: "ut dolor non eiusmod est culpa quis voluptate quis magna officia eu in tempor non fugiat tempor irure cupidatat velit qui adipiscing pariatur cupidatat ut fugiat qui ullamco est sunt sit irure occaecat deserunt voluptate reprehenderit minim labore ullamco ullamco minim dolore fugiat ullamco magna quis veniam anim eiusmod laboris Duis culpa fugiat Excepteur labore laborum mollit qui ea consequat tempor tempor ut aute amet cupidatat ea occaecat sunt exercitation tempor cupidatat Duis aliquip ea cupidatat velit eiusmod id id dolore fugiat eiusmod id Excepteur incididunt id labore dolore eiusmod"
  },
  {
    id: 4,
    title: "AC-GAN",
    aurthor: "Mirza & Osindero, 2014",
    image: type4,
    copy: "ut dolor non eiusmod est culpa quis voluptate quis magna officia eu in tempor non fugiat tempor irure cupidatat velit qui adipiscing pariatur cupidatat ut fugiat qui ullamco est sunt sit irure occaecat deserunt voluptate reprehenderit minim labore ullamco ullamco minim dolore fugiat ullamco magna quis veniam anim eiusmod laboris Duis culpa fugiat Excepteur labore laborum mollit qui ea consequat tempor tempor ut aute amet cupidatat ea occaecat sunt exercitation tempor cupidatat Duis aliquip ea cupidatat velit eiusmod id id dolore fugiat eiusmod id Excepteur incididunt id labore dolore eiusmod"
  }
];

function determineClasses(indexes, cardIndex) {
  if (indexes.currentIndex === cardIndex) {
    return "active";
  } else if (indexes.nextIndex === cardIndex) {
    return "next";
  } else if (indexes.previousIndex === cardIndex) {
    return "prev";
  }
  return "inactive";
}

const CardCarouselSecond = () => {
  const [indexes, setIndexes] = useState({
    previousIndex: 0,
    currentIndex: 0,
    nextIndex: 1
  });

  const handleCardTransition = useCallback(() => {
    if (indexes.currentIndex >= cardItems.length - 1) {
      setIndexes({
        previousIndex: cardItems.length - 1,
        currentIndex: 0,
        nextIndex: 1
      });
    } else {
      setIndexes(prevState => ({
        previousIndex: prevState.currentIndex,
        currentIndex: prevState.currentIndex + 1,
        nextIndex:
          prevState.currentIndex + 2 === cardItems.length
            ? 0
            : prevState.currentIndex + 2
      }));
    }
  }, [indexes.currentIndex]);

  return (
    <div className="containerS">
      <ul className="card-carouselS">
        {cardItems.map((card, index) => (
          <div
            key={card.id}
            className={`cardS ${determineClasses(indexes, index)}`}
            onClick={handleCardTransition}
          >
            <div className="left">
              <h2>{card.title}</h2>
              <p>{card.copy}</p>
              <p>(- {card.aurthor})</p>
            </div>
          
            <img src={card.image} className="right" />
          </div>
        ))}
      </ul>
    </div>
  );
};

export default CardCarouselSecond;