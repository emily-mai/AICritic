import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import model

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
application = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.config["suppress_callback_exceptions"] = True


app.layout = html.Div(
    "hello world"
)


if __name__ == "__main__":
    # Run on docker
    application.run(host="0.0.0.0", port=80, debug=True)

    # Run locally
    # app.run_server(port=port, debug=True)

    text = '''
    Diana Prince lives quietly among mortals in the vibrant, sleek 1980s -- an era of excess driven by the pursuit of having it all. Though she's come into her full powers, she maintains a low profile by curating ancient artifacts, and only performing heroic acts incognito. But soon, Diana will have to muster all of her strength, wisdom and courage as she finds herself squaring off against Maxwell Lord and the Cheetah, a villainess who possesses superhuman strength and agility.
    '''
    prediction = model.predict_plot_rating(text)
    print("Text: {}".format(text))
    print("Prediction: {}".format(prediction))
