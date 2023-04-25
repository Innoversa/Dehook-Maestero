import React, { useState, useRef } from "react";
import "./comp.css";

const ReadOutLoudButton = ({ text }) => {
  const [isReading, setIsReading] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const utteranceRef = useRef(null);

  const handleReadOutLoud = () => {
    const speech = new SpeechSynthesisUtterance(text);
    utteranceRef.current = speech;
    setIsReading(true);
    setIsPaused(false);
    speech.onend = () => {
      setIsReading(false);
    };
    window.speechSynthesis.speak(speech);
  };

  const handleStopReading = () => {
    window.speechSynthesis.cancel();
    setIsReading(false);
    setIsPaused(false);
  };

  const handlePauseReading = () => {
    window.speechSynthesis.pause();
    setIsPaused(true);
  };

  const handleResumeReading = () => {
    window.speechSynthesis.resume();
    setIsPaused(false);
  };

  return (
    <div>
      <button
        className="read-out-loud"
        disabled={isReading}
        onClick={handleReadOutLoud}
      >
        {isReading ? "Reading..." : "Read Email Content"}
      </button>
      {isReading && (
        <div>
          {isPaused ? (
            <button className="read-out-loud" onClick={handleResumeReading}>
              Resume
            </button>
          ) : (
            <button className="read-out-loud" onClick={handlePauseReading}>
              Pause
            </button>
          )}
          <button className="read-out-loud" onClick={handleStopReading}>
            Stop
          </button>
        </div>
      )}
    </div>
  );
};

export default ReadOutLoudButton;
