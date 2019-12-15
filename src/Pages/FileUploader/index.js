import React, { Component } from "react";
import "./index.css";
import Upload from "./upload/Upload";
import NavBar from "../../Components/navbar";

class UploadPage extends Component {
  render() {
    return (
      <div className="">
        <NavBar className="mb-5" /> 
        <div className="Page">
          <div className="card mt-3 mt-md-4 p-3 col-10 col-md-9 col-lg-8 col-xl-7">
            <Upload />
          </div>
        </div>
      </div>
    );
  }
}

export default UploadPage;
