import React, { useState } from "react";
import "./comp.css";
function FraudDetector({ inputString }) {
  const [score, setScore] = useState(1);
  const [sentence, setSentence] = useState("Looks like a fake email");

  const handleSubmit = async (event) => {
    event.preventDefault();

    const response = await fetch("/fraud_detector", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: inputString }),
    });

    const data = await response.json();
    setScore(data.score);
    setSentence(data.sentence);
  };

  return (
    <div className="score">
      Fraud Score: {score}
      <br />
      Reason: {sentence}
    </div>
  );
}

export default FraudDetector;
