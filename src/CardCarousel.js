import React, { useEffect, useCallback, useState } from "react";
import "./CardCarousel.css";


const cardItems = [
  {
    id: 1,
    title: "Stacked Card Carousel",
    copy:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus sit amet dui scelerisque, tempus dui non, blandit nulla. Etiam sed interdum est."
  },
  {
    id: 2,
    title: "Second Item",
    copy: "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
  },
  {
    id: 3,
    title: "A Third Card",
    copy:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus sit amet dui scelerisque, tempus dui non, blandit nulla."
  },
  {
    id: 4,
    title: "Fourth",
    copy: "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
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

const CardCarousel = () => {
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
    <div className="container">
      <ul className="card-carousel">
        {cardItems.map((card, index) => (
          <div
            key={card.id}
            className={`card ${determineClasses(indexes, index)}`}
            onClick={handleCardTransition}
          >
            <li>
              <h2>{card.title}</h2>
            </li>
            <p>{card.copy}</p>
          </div>
        ))}
      </ul>
    </div>
  );
};

export default CardCarousel;