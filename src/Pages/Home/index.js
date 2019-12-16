import React from 'react';
import NavBar from "../../Components/navbar";
import AppConfig from "../../Configs/package";
import ImageExplorer from "../../Components/imageExplorer";
import SelectCtrl from "../../Controls/Select";


class  Home  extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            images : {
                title: "Generate init images",
                list : "init"
            }
        }
        this.onSelectChange = this.onSelectChange.bind(this)
    }
    getImages(){
        let imgs = [];
        let init_images = [];
        // let path = "/srgan-investigation/samples/";
        let path = "/samples/";

        for(let i = 10; i < 630; i += 10){
            if (i < 100){
                init_images.push( path + "train_g_init_" + i + ".png");
            }
            imgs.push( path + "train_g_"+i + ".png");
        }
        return {
            "images" : imgs,
            "init"  :   init_images
        }
    }
    onSelectChange(e){
        let title = this.state.images.title;

        for (let i = 0; i < e.target.options.length; ++i){
            if(e.target.options[i].value === e.target.value){
                title = e.target.options[i].text
                break
            }
        }
        let images = this.state.images;
        images.list = e.target.value
        images.title = title;

        this.setState({
            images : images
        })
    }
    render(){
        let images = this.getImages()[this.state.images.list]
        
        let options = [
            {
                label : "Generate init images",
                value : "init"
            },
            {
                label : "Generate-Adversarial images",
                value : "images"
            }
        ]
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
                            <div className="col-4">
                                <SelectCtrl name="select-view"  caption="Select image set" onChange={this.onSelectChange} value={this.state.images.list} options={options}/>
                            </div>
                            <div className="col-12 px-0">
                                <ImageExplorer list={images} title={this.state.images.title}/>
                            </div>
                        </div>                                            
                    </div>
                </div>
            </div>      
        );
    }    
  }

  export default Home;
  