import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def predict(symbol: str):
    # Fetch historical data for Goldman Sachs (GS)
    gs = yf.Ticker(symbol).history(period="max")

    # Remove unnecessary columns

    if "Dividends" in gs.columns:
        del gs["Dividends"]
    if "Stock Splits" in gs.columns:
        del gs["Stock Splits"]


    # Create a target column indicating whether the stock price will go up or down
    gs["Tomorrow"] = gs["Close"].shift(-1)
    gs["Target"] = (gs["Tomorrow"] > gs["Close"]).astype(int)

    # Define predictors
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    # Initialize and train the RandomForestClassifier model
    model = RandomForestClassifier(
        n_estimators=200, min_samples_split=100, random_state=1
    )
    train = gs.iloc[:-100]
    model.fit(train[predictors], train["Target"])

    # Make predictions on the test data
    test = gs.iloc[-100:]
    preds = model.predict(test[predictors])

    # Calculate precision score
    precision = precision_score(test["Target"], preds)

    # Determine if the majority of predictions suggest the stock price will go up or down
    if (preds.sum() / len(preds)) > 0.5:
        prediction = "ACCORDING TO THE PREDICTION THE STOCK PRICE WILL GO UP"
    else:
        prediction = "ACCORDING TO THE PREDICTION THE STOCK PRICE WILL FALL DOWN"

    print(f"Precision Score: {precision}")
    print(f"The majority of predictions suggest the stock price will go {prediction}.")
    return prediction


@app.get("/predict")
def predict_view():
    symbol = request.args.get("symbol")
    if not symbol:
        return "Symbol not provided", 400
    return predict(symbol), 200


if __name__ == "__main__":
    app.run()