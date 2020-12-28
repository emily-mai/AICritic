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
    [
        dbc.Textarea(className="mb-3", bs_size="lg", placeholder="Please enter a description...", id="text-input"),
        dbc.Button("Predict Rating", id="submit-button", className="btn btn-primary btn-lg btn-block")
    ]
)


@app.callback(
    Output("rating-output", "children"),
    [Input("submit-button", "n_clicks")],
    [State("text-input", "value")]
)
def predict_rating(n_clicks, text):
    if n_clicks is None:
        return "Not clicked."
    else:
        return f"Clicked {n_clicks} times."


if __name__ == "__main__":
    application.run(debug=True)
