import React, {Component} from "react";
import ImageCtrl from "../../Controls/Image";


class ImageExplorer extends Component {
    constructor(props){
        super(props);
        
        this.state = {

        }
    }

    render(){
        return <div className="col-12">
                    <h4 className="">
                        {this.props.title}
                    </h4>
                </div>;
    }
}

export default ImageExplorer;