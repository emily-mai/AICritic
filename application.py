import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import model

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
application = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.config["suppress_callback_exceptions"] = True


app.layout = html.Div(
    [
        html.Div(id="rating-output"),
        dbc.Textarea(
            className="mb-3",
            placeholder="Please enter a 1-3 paragraph plot summary ...",
            id="text-input",
            style={"height": "500px"}
        ),
        dbc.Button("Predict Rating", id="submit-button", className="btn btn-primary btn-lg btn-block")
    ],
    style={
        "margin-top": "20%",
        "margin-left": "20%",
        "margin-right": "20%",
        "margin-bottom": "5%",
    }
)


@app.callback(
    Output("rating-output", "children"),
    [Input("submit-button", "n_clicks")],
    [State("text-input", "value")]
)
def predict_rating(n_clicks, text):
    if n_clicks is None or text is None:
        return
    else:
        prediction = model.predict_plot_rating(text)
        print("Text: {}".format(text))
        print("Prediction: {}".format(prediction))
        if 0 <= prediction < 4:
            message = "That's rubbish! Better luck next time mate."
            color = "danger"
        elif 4 <= prediction < 7:
            message = "Very mediocre..."
            color = "warning"
        else:
            message = "Oi! We've got a Steve Spielberg innit!"
            color = "success"
        alert = dbc.Alert(
            [
                html.H1(str(round(prediction, 1)) + "/10"),
                html.P(message),
                dbc.Progress(str(int(prediction * 10)) + "%", value=prediction * 10, color="info")
            ],
            color=color
        )
        return alert


if __name__ == "__main__":
    # Run on docker
    application.run(host="0.0.0.0", port=80, debug=True)

    # Run locally
    # app.run_server(debug=True)

