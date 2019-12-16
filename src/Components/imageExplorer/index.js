import React, {Component} from "react";
import ImageCtrl from "../../Controls/Image";


class ImageExplorer extends Component {
    constructor(props){
        super(props);
        
        this.state = {
            active : 0,
            interval : 5000,
            list : this.props.list
        }
    }
    componentDidMount(){
        this.updateImage();
    }

    componentDidUpdate(oldProps){
        if(oldProps.list !== this.props.list){
            this.setState({
                list : this.props.list
            })
        }
    }

    updateImage(){
        let interval = setInterval(() => {
            let active = this.state.active;

            if(active === this.state.list.length - 1){
                active = -1;
            }
            active += 1;

            this.setState({
                active : active
            })
        }, this.state.interval);

        return interval
    }

    showCarousel(){
        let img = this.state.list[this.state.active];
        let name = img.replace(".png", "").replace(".jpg", "").split("/").pop();
        
        return <ImageCtrl name={name} src={img} />
    }
    render(){
        return <div className="col-12 px-0">
                    <h4 className="col-12">
                        {this.props.title}
                    </h4>
                    <div className="col-12 image-review px-0">
                        {this.showCarousel()}
                    </div>
                </div>;
    }
}

export default ImageExplorer;