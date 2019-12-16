import React, { Component } from 'react';
import { Router, Route } from 'react-router-dom';
import { history } from './helpers/history';
import PrivateRoute from "./Components/privateRoute"


import UploadPage from "./Pages/FileUploader";
import Home from "./Pages/Home";
import Login from "./Pages/Auth/Login";

import './Css/bootstrap.css';
import './Css/App.css';



class App extends Component {
  render() {
    return (
      <Router history={history}>     
          <Route exact strict path="/login" component={Login} />
          <Route exact strict path="/" component={Home} />
          <PrivateRoute exact strict path="/test" component={UploadPage} />
          <Route exact strict path="/srgan-investigation/login" component={Login} />
          <Route strict path="/srgan-investigation" component={Home} />
          <PrivateRoute exact strict path="/srgan-investigation/test" component={UploadPage} />
      </Router>
    );
  }
}

export default App;
