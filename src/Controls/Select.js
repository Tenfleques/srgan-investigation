import React from 'react';

const Select = (props) => {
    return (
        <div className="form-group">
            <label htmlFor={props.name}>{props.caption}</label>
            <select 
                className="form-control" 
                onChange={props.onChange}
                id={props.name}
                name={props.name}
                defaultValue={props.value}
                >
                {
                    props.options.map((v,i) => {
                        return <option key={i} value={v.value}>{v.label}</option>
                    })
                }
            </select>
        </div>
    );    
}
export default Select
  
  