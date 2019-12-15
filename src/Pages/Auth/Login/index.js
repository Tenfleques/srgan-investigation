import React from 'react';
import { Redirect } from 'react-router-dom'
import GithubAPI from "../../../helpers/gh-helper"
import Application from "../../../Configs/package"
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner, faStop} from '@fortawesome/free-solid-svg-icons'

class LoginPage extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            login: '',
            password: '',
            submitted: false,
            user : {},
            error: false
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }
    componentDidMount(){
        sessionStorage.removeItem('user');
    }

    handleChange(e) {
        const { name, value } = e.target;
        this.setState({ [name]: value });
    }

    handleSubmit(e) {
        e.preventDefault();
        const { login, password } = this.state;
        this.setState({submitted: true})
        let user = {
            "name" : login,
            "token" : password
        }
        const api = new GithubAPI({
            token: user.token
        });

        api.setRepo(user.name, Application.repo);

        let attempt_set_api = api.setBranch(Application.branch);

        attempt_set_api.then(() => {
            sessionStorage.setItem("user", JSON.stringify(user));
            this.setState({ submitted: true, user : user });
        })

        attempt_set_api.catch((e) => {
            this.setState({error: true})
        })
    }

    renderLoginForm() {
        const { login, password, submitted, error } = this.state;
        return (
            <div className="container p-5">
                <form className="row mt-5 p-md-5 bg-primary text-white max-w-600px mx-auto" onSubmit={this.handleSubmit}>
                    <div className={error ? "col-12 text-danger": "invisible col-12"}>
                        Пользователь не аутентифицирован
                    </div>
                    <div className='form-group col-12'>
                        <label htmlFor="login">логин</label>
                        <input
                            type="text"
                            name="login"
                            value={login}
                            className={"form-control col-12 mb-3" + (submitted && !login ? ' border-danger' : '')}
                            onChange={this.handleChange}
                        />
                        {submitted && !login &&
                            <div className="text-danger">обязателная поля</div>
                        }
                    </div>
                    <div className='form-group col-12'>
                        <label htmlFor="password">пароль</label>
                        <input
                            type="password"
                            name="password"
                            value={password}
                            className={"form-control col-12 mb-3" + (submitted && !password ? ' border-danger' : '')}
                            onChange={this.handleChange}
                        />
                        {submitted && !password &&
                            <div className="text-danger">обязателная поля</div>
                        }
                    </div>
                    <div className="form-group col-12">
                        <button 
                            type="submit"
                            className="form-control col-12 mb-3"
                        >
                            Войти &nbsp;
                            {
                                submitted ?
                                <FontAwesomeIcon icon={faStop} className={error?
                                "text-danger d-none" : "d-none"} /> : ""
                            }
                            {
                                submitted ?
                                <FontAwesomeIcon icon={faSpinner} className={error?
                                    "d-none" : "text-success"} /> : ""
                            }
                                
                        </button>
                        
                    </div>
                </form>
            </div>
        );
    }

    render() {
        return this.state.user.token
            ? <Redirect to={{ pathname: '/', state: { from: "/login" } }} />
            : this.renderLoginForm();
    }
}

export default LoginPage