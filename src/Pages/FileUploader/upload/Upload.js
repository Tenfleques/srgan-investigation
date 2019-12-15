import React, { Component } from "react";
import Dropzone from "../dropzone/Dropzone";
import "./Upload.css";
import GithubAPI from "../../../helpers/gh-helper"
import Application from "../../../Configs/package"
import AutoCompleteTextBox from "../../../Controls/AutocompleteTextBox"
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner, faCheckCircle } from '@fortawesome/free-solid-svg-icons'




class Upload extends Component {
  constructor(props) {
    super(props);
    
    this.state = {
      files: [],
      uploading: false,
      uploadProgress: {},
      successfullUploaded: false,
      gh_api : {},
      catalog: "",
      authorized: false,
      user: {}
    };

    this.onFilesAdded = this.onFilesAdded.bind(this);
    this.uploadFiles = this.uploadFiles.bind(this);
    this.onPathChanged = this.onPathChanged.bind(this);
    this.renderActions = this.renderActions.bind(this);
  }
  componentDidMount(){
    let user = sessionStorage.getItem("user");
    if(user){
      user = JSON.parse(user);

      const api = new GithubAPI({
        token: user.token
      })  
      api.setRepo(user.name, Application.repo);

      let attempt_set_api = api.setBranch(Application.branch);

      attempt_set_api.then(() => {
          this.setState({ authorized: true, user : user, gh_api: api });
      })

      attempt_set_api.catch((e) => {
        this.setState({ authorized: false, user : user });
      })
    }
  
  }
  onFilesAdded(files) {
    this.setState(prevState => ({
      files: prevState.files.concat(files)
    }));
  }

  onPathChanged(ev) {
    let path = ev.target.value;

    if(path[path.length - 1] !== "/"){
      path += "/"
    }
    if(path[0] === "/"){
      path = path.slice(1);
    }

    this.setState({catalog: path});
  }
  async uploadFiles() {
    this.setState({ uploadProgress: {}, uploading: true });
    let api = this.state.gh_api
    const promises = [];
  
    for(let i = 0; i < this.state.files.length; ++i){
  
      let fnam = this.state.files[i].name

      promises.push(new Promise((resolve, reject) => {
        var fr = new FileReader();  
        fr.onload = () => {
          resolve({ name: fnam, path: Application.path + this.state.catalog + fnam, content:  fr.result } )
        };
        fr.addEventListener('progress', (event)=>{
          if (event.lengthComputable) {
            const copy = { ...this.state.uploadProgress };
            copy[fnam] = {
              state: "pending",
              percentage: (event.loaded / event.total) 
              * 100
            };
            this.setState({ uploadProgress: copy });
          }
        });
        fr.readAsText(this.state.files[i], "UTF-8"); 
      }));
    }

    try {
      let that = this;
      Promise.all(promises).then(function(files){
        let filenames = files.map(f => f.name);
        let upload = api.pushFiles("file uploads: " + filenames.join(", "), files);        

        let updateLoaded = (f) => {
          const copy = { ...that.state.uploadProgress };
          copy[f] = { state: "done", percentage: 100 };
          that.setState({ uploadProgress: copy });           
        }

        upload.then((res) => {
          if(res){
            if(res.status === 200){
                for (var i = 0; i < filenames.length; ++i){
                  setTimeout(updateLoaded(filenames[i]), Math.random()*10000)
                }
              }
          }          
        })
        upload.catch((e) => {
          for (var i = 0; i < filenames.length; ++i){
            const copy = { ...that.state.uploadProgress };
            copy[filenames[i]] = { state: "fail", percentage: 100 };
            that.setState({
              successfullUploaded : false,
              uploadProgress: copy
            })
          }
        })
      });
      

      this.setState({ successfullUploaded: true, uploading: false });
    } catch (e) {
      this.setState({ successfullUploaded: true, uploading: false });
    }
  }

  renderProgress(file) {
    const uploadProgress = this.state.uploadProgress[file.name];
    if (this.state.uploading || this.state.successfullUploaded) {
      return (
        <span className="d-inline pl-3">
            <FontAwesomeIcon icon={faCheckCircle} className={uploadProgress && uploadProgress.state === "done"?
            "text-secondary" : "d-none" } />
             <FontAwesomeIcon icon={faSpinner} className={uploadProgress && uploadProgress.state === "done"?
          "d-none" : "text-warning fa-spin"} />                   
        </span>
      );
    }
  }

  renderActions() {
    if (this.state.successfullUploaded) {
      return (
        <button
          onClick={() =>
            this.setState({ files: [], successfullUploaded: false })
          }
        >
          Очистить
        </button>
      );
    } else {
      return (
        <button
          disabled={this.state.files.length < 0 || this.state.uploading}
          onClick={this.uploadFiles}
          className="px-5"
        >
          Загрузить
        </button>
      );
    }
  }

  render() {
    if (this.state.authorized){  
      return (
        <div className="container-fluid">
          <div className="col-12">
            <span className="Title py-3">Загрузка файлов</span>
            <div className="row pt-3">
              <AutoCompleteTextBox 
                name="auto-complete-catalogs" 
                caption="каталог файлов" 
                className="col-12 px-0 border-bottom form-group text-left py-3 mb-3" 
                help="Определить в какую группу отправить файлы. e.g. vsuet/vosdux" 
                placeholder="" 
                onChange={this.onPathChanged}
                />
              <div className="col-12 col-md-8 col-lg-6"> 
                <Dropzone
                  onFilesAdded={this.onFilesAdded}
                  disabled={this.state.uploading || this.state.successfullUploaded}
                />
                <div className="col-12 text-center py-3">{this.renderActions()}</div>
              </div>
              <div className="col-12 col-md-4 col-lg-6">
                <div className="row text-left">
                  {this.state.files.map(file => {
                    return (
                      <div key={file.name} className="col-12 col-lg-6">
                        <span className="">{file.name}</span>
                        {this.renderProgress(file)}
                      </div>
                    );
                  })}
                </div>            
              </div>
            </div>
            
          </div>
        </div>
      );
    }
    return "загрузка ....";
  }
}

export default Upload;
