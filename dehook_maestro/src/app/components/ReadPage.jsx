import React, { useState } from "react";
import "./comp.css";

const handleClick = () => {
  // handle button click here
};

function HoverButton({ label, tooltip }) {
  const [showTooltip, setShowTooltip] = useState(false);

  const handleMouseEnter = () => {
    setShowTooltip(true);
  };

  const handleMouseLeave = () => {
    setShowTooltip(false);
  };

  return (
    <button
      onClick={handleClick}
      className="hover-button"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {label}
      {showTooltip && <span className="sr-only">{tooltip}</span>}
    </button>
  );
}

export default HoverButton;
