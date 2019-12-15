import React from 'react';
import NavBar from "../../Components/navbar";
import AppConfig from "../../Configs/package";
import ImageExplorer from "../../Components/imageExplorer";


class  Home  extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            images : {
                title: "Stock Cyclic",
                list : []
            }
        }
    }
    getImages(){
        
    }
    render(){
        return (
            <div className="">
                <NavBar className="mb-5" />            
                <div className="container-fluid mt-3">
                    <div className="col-12 h3">
                        Welcome to {AppConfig.appname} 
                    </div>
                    <div className="col-12">
                        {AppConfig.description} 
                    </div>
                    <div className="container-fluid">
                        <div className="row">
                            <div className="col-8">
                                <ImageExplorer title={this.state.images.title}/>
                            </div>
                        </div>                                            
                    </div>
                </div>
            </div>      
        );
    }    
  }

  export default Home;
  