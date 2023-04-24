"use client"
import Image from 'next/image'
import { Inter, Atkinson_Hyperlegible } from 'next/font/google'
import './globals_dark.css'
import React, { useState } from "react"
import ReactDOM from "react-dom/client"

const Atkinson = Atkinson_Hyperlegible({subsets:['latin'], weight:"400"})

function handleClick() {
  console.log(document.getElementsByClassName("text_input1"))
}

class EssayForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      value: "New Entry Here"
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({value: event.target.value});
  }

  handleSubmit(event) {
    console.log(this.state.value);
    // alert('An essay was submitted: ' + this.state.value);
    event.preventDefault();
  }

  render() {
    return (
      <div>
        <form onSubmit={this.handleSubmit} id='form1'>
          <textarea rows="10" cols="40" value={this.state.value} onChange={this.handleChange} />
        </form>
        <input className="submit_button" type="submit" value="Submit" form="form1"/>
      </div>
    );
  }
}


export default function Home() {
  // const root = ReactDOM.createRoot(document.getElementById('root'));
  const [file, setFile] = useState(null);
  const [createObjectURL, setCreateObjectURL] = useState(null);
  const uploadToClient = (event) => {
    if (event.target.files && event.target.files[0]) {
      const i = event.target.files[0];
      setFile(i);
      setCreateObjectURL(URL.createObjectURL(i));
    }
  };
  const uploadToServer = async (event) => {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/file", {
      method: "POST",
      body
    });
  };
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">

      <div className="main_logo">
        <Image
          className="relative dark:invert"
          src="/fish_n_hook.webp"
          alt="dehook maestro Logo"
          width={300}
          height={10}
          priority
        />
        <div className="main_tital">
          <h1 className={`${Atkinson.className}`}>
            Dehook Maestro
          </h1>
        </div>
      </div>

      <div className="rowC">
        <div>
          <h2 className={`${Atkinson.className} mb-1 font-semibold`}>
            Submit your file 
          </h2>
          <input type="file" name="myImage" onChange={uploadToClient} />
          <div>          
            <button className="submit_button_1"
            type="submit"
            onClick={uploadToServer}
            >
            Submit
            </button></div>
        </div>

        <div>
          <h2 className={`${Atkinson.className} mb-1 font-semibold`}>
            Type or copy & paste your email contents below
          </h2>
          <EssayForm />
        </div>
      </div>
    </main>
  )
}