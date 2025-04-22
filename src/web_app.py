import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import joblib

# Load data and model
data = pd.read_csv("_data/cleaned_test_data.csv")  # Adjust path if needed
model = joblib.load("artifacts/random_forest_model.pkl")  # or xgb_model.pkl

X = data.drop("GradeClass", axis=1)
y_true = data["GradeClass"]
predictions = model.predict(X)

grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
y_true_labels = y_true.map(grade_map)
y_pred_labels = pd.Series(predictions).map(grade_map)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("BrightPath Student Grade Predictor", className="text-center mt-4 mb-4"),
    
    html.P("This dashboard shows the predictions made by the Random Forest model on test data."),

    dbc.Row([
        dbc.Col([
            html.H5("Sample Prediction Table:"),
            dcc.Graph(
                figure={
                    'data': [{
                        'type': 'table',
                        'header': {'values': list(X.head().columns) + ['Actual', 'Predicted']},
                        'cells': {
                            'values': [X[col].head().tolist() for col in X.columns] + 
                                      [y_true_labels.head().tolist(), y_pred_labels.head().tolist()]
                        }
                    }]
                }
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Model Performance:"),
            html.Div([
                html.P(f"Accuracy: {(y_true == predictions).mean():.2f}"),
                html.Img(src="/assets/RandomForest_Confusion-matrix_Heat.png", style={"width": "100%"})
            ])
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
