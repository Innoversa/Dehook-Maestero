import React, { useState } from "react";

const ReadOutLoudButton = ({ text }) => {
  const [isReading, setIsReading] = useState(false);

  const handleReadOutLoud = () => {
    const speech = new SpeechSynthesisUtterance(text);
    setIsReading(true);
    speech.onend = () => {
      setIsReading(false);
    };
    window.speechSynthesis.speak(speech);
  };

  return (
    <button disabled={isReading} onClick={handleReadOutLoud}>
      {isReading ? "Reading..." : "Read Out Loud"}
    </button>
  );
};

export default ReadOutLoudButton;
