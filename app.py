# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

model_path = 'scaler.pkl'
with open(model_path, 'rb') as file:
    scaler = pickle.load(file)
    
model_path = 'oe.pkl'
with open(model_path, 'rb') as file:
    oe = pickle.load(file)
    
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    try:
        features = [str(x) for x in request.form.values()]
        bedr=int(features[0])
        bathr=int(features[1])
        floors=int(features[2])
        year=int(features[3])
        locat=features[4]
        cond=features[5]
        gar=features[6]
        ar=int(features[7])
        
        num = scaler.transform([[ar]])
        cat = oe.transform([[locat,cond,gar]])
        
        final_features = np.array([[bedr,bathr,floors,year,cat[0,0],cat[0,1],cat[0,2],num[0,0]]])
        
        # Make prediction
        prediction = model.predict(final_features)
        output = round(0.637*prediction[0],2)

        return render_template('index.html', prediction_text='Prediction: US${}'.format(output))
    
    except:
        return render_template('index.html', prediction_text='Error!! Enter Values Again.\n\n Possible Errors:\n1. Values entered are wrong.')

if __name__ == "__main__":
    app.run(debug=True)