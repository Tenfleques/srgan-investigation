import React from 'react';
import { connect } from 'react-redux';
import { userActions } from '../../redux';

class EnvBar extends React.Component {
    constructor(props) {
        super(props);            
        this.state = {
            nose: {},
            model: {},
            session: false,
            collapsed: false,
            time: new Date(), //todo turn into time diff from time of loging in,
            user: {} 
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleCollapseForm = this.handleCollapseForm.bind(this);
    }
    handleChange(e) {
        const { name, value } = e.target;
        this.setState({ [name]: value });
    }
    handleCollapseForm(){
      this.setState({
        collapsed: !this.state.collapsed
      })
    }
    handleSubmit(e) {
        e.preventDefault();
        this.setState({ submitted: true });
        const { title, description } = this.state;
        const { dispatch } = this.props;
        if (title && description) {
            let res = userActions.addTodo(title, description)
            dispatch(res);
            console.log(res)
            /*res.then(r => this.setState({
                title: '',
                description: '',
                submitted: false
            }))*/
            
        }
    }

    render() {
        //const { title, description, submitted } = this.state;
        return (
            <div className="col-12">
                <h4> Глобальные параметры </h4>
            </div>
        );
    }
}

function mapStateToProps(state) {
    return {
        state
    };
}
export default connect(mapStateToProps)(EnvBar)