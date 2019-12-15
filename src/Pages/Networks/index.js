import React, {Component} from 'react';
import { connect } from 'react-redux';
import ModelsList from "../../Components/lists/ModelsList"
import EnvCard from "../../Components/env"

class App extends Component {
  constructor(props) {
      super(props);
  }
  render() {
    return (
        <div className="row mt-5">
          <div className="d-none d-md-block col-md-4 col-lg-3">
            <EnvCard/>
          </div>
          <div className="col-12 col-md-8 col-lg-9">
            <ModelsList/>
          </div>
        </div>
    );
  }
}

function mapStateToProps(state) {
  const { alert, model } = state;
  return {
      alert,
      model
  };
}

export default connect(mapStateToProps)(App);
