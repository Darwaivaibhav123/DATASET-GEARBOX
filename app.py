from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the machine learning model and selected feature columns
selected_features = ['S1_F3', 'S1_F4', 'S1_F9', 'S1_F10', 'S1_F11', 'S1_F13', 'S2_F7', 'S2_F8', 'S2_F9', 'S2_F10']
data = pd.read_csv(r"C:\Users\DELL\Desktop\imag\Processed_final_data.csv")  # Update file path
threshold = 0  # Set your threshold based on the actual data distribution
data['Fault'] = np.where(data['Fault'] > threshold, 1, 0)  # Use 1 for 'Broken' and 0 for 'Healthy'
selected_data = data[['Fault'] + selected_features]
X_train, _, y_train, _ = train_test_split(selected_data.drop("Fault", axis=1), selected_data["Fault"], test_size=0.33, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_rf.fit(X_train, y_train)

# Adjust threshold based on model performance on validation dataset
# You can experiment with different threshold values to minimize false positives
# For example, if false positives are more critical, you might want to increase the threshold
# Conversely, if false negatives are more critical, you might want to decrease the threshold
# You can adjust this threshold based on the model's performance on a validation dataset
threshold = 0.5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        vibration_data = [float(request.form.get('vibrationInput{}'.format(i))) for i in range(1, 11)]
        prediction_proba = clf_rf.predict_proba([vibration_data])[0][1]  # Probability of being "Fault detected"
        prediction = 1 if prediction_proba >= threshold else 0
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
