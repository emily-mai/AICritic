import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
application = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.config["suppress_callback_exceptions"] = True


app.layout = html.Div(
    "hello world"
)


if __name__ == "__main__":
    application.run(debug=True, host="0.0.0.0", port="80")
