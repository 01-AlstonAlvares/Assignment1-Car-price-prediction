import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import joblib, json

# === Load model & metadata ===
model = joblib.load("app/deploy_assets/pipeline.joblib")
with open("app/deploy_assets/assets.json", "r") as f:
    meta = json.load(f)

features = meta["features"]
num_cols = meta["num_cols"]
cat_cols = meta["cat_cols"]

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Car Price Prediction"

# Build inputs
input_components = []
for i, col in enumerate(features):
    input_components.append(
        html.Div([
            html.Label(col, style={'fontWeight': '600'}),
            dcc.Input(id=f"input-{i}", type='text',
                      placeholder=f"Enter {col} (optional)",
                      style={'width': '60%'})
        ], style={'margin-bottom': '6px'})
    )

app.layout = html.Div([
    html.H1("Car Price Prediction System", style={'textAlign':'center'}),
        # === User Guide Section ===
    html.Div([
        html.H2("How to Use This App", style={'marginBottom': '10px'}),
        html.Ul([
            html.Li("1. Enter details of the car in the input fields provided."),
            html.Li("2. Leave any field blank if you don’t know the value."),
            html.Li("3. Once all available details are filled, click the 'Predict Price' button."),
            html.Li("4. The predicted price will appear below the button."),
            html.Li("Note: Predictions are based on historical data and may not reflect the exact market value.")
        ], style={'lineHeight': '1.8'})
    ], style={
        "backgroundColor": "#f9f9f9",
        "padding": "15px",
        "borderRadius": "10px",
        "marginBottom": "20px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"
    }),

    # === Input Section ===
    html.Div([
        html.H3("Enter Car Features"),
        html.Div(input_components),
        html.Button("Predict Price", id="predict-button", n_clicks=0),
        html.Div(id="prediction-output", style={"margin-top":"12px", "fontWeight":"bold"})
    ], style={"width":"70%", "margin":"0 auto"})
])

state_args = [State(f"input-{i}", "value") for i in range(len(features))]

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    *state_args
)
def predict(n_clicks, *values):
    if not n_clicks:
        return ""
    try:
        vals = [v if (v is not None and str(v).strip() != "") else np.nan for v in values]
        input_df = pd.DataFrame([dict(zip(features, vals))])

        # numeric conversion
        for c in num_cols:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce")
        for c in cat_cols:
            input_df[c] = input_df[c].fillna("missing")

        input_df = input_df.reindex(columns=features, fill_value=np.nan)

        pred_log = model.predict(input_df)[0]
        pred_price = np.exp(pred_log)
        return f"Predicted Car Price: ₹{pred_price:,.2f}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
