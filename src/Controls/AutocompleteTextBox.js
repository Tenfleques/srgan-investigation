import React, { Component } from 'react';

class AutoCompleteTextBox extends Component {
    constructor(props) {
        super(props);
        this.autoComplete = this.autoComplete.bind(this)
    }
    
    autoComplete(e){
        
    }
    render () {
        return (
            <div className={this.props.className}>
                <label htmlFor={this.props.name}>{this.props.caption}</label>
                <input 
                    type = "text"
                    className="form-control" 
                    onInput={this.autoComplete}
                    onChange={this.props.onChange}
                    id={this.props.name}
                    name={this.props.name}
                    placeholder={this.props.placeholder}                
                    />
                <span className="muted small">
                    {this.props.help}
                </span>
            </div>
        );
    }        
}
export default AutoCompleteTextBox
  
  