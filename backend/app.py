# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load the model
# model = joblib.load('heart_attack_predictor.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # Get the JSON data from the request
#     df = pd.DataFrame(data)
    
#     prediction = model.predict(df)
#     output = {'predictions': prediction.tolist()}
    
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load the trained model
model = joblib.load('heart_attack_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert JSON to DataFrame
    df = pd.DataFrame(data)
    # Ensure the DataFrame has the correct columns
    expected_columns = ['age', 'thalach', 'trtbps_winsorize', 'oldpeak_winsorize_sqrt', 'sex_1', 
                        'cp_1', 'cp_2', 'cp_3', 'exang_1', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 
                        'ca_3', 'ca_4', 'thal_2', 'thal_3']
    df = df[expected_columns]
    # Make predictions
    predictions = model.predict(df)
    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)

