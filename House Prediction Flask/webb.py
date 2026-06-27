import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

model=joblib.load('decision_m.pkl')
city_classes={0:'delhi',1:'bombay',2:'jaipur'}

web=Flask(__name__)
@web.route('/')

def home(): #this is for getting the first page is predict page of html
    return render_template('index.html')
@web.route('/predict',methods=['POST'])

# now we create the predict function which contains the prediction part of the model
def predict():
    try:
        data=request.json
        cityy=data['features']
        prediction=model.predict([cityy])[0]
        if isinstance(prediction, str):
            city_name = prediction
            prediction = list(city_classes.keys())[list(city_classes.values()).index(prediction)]
        else:
            prediction = int(prediction)
            city_name = city_classes[prediction]
        return jsonify({'prediction': prediction, 'city':city_name})

    except Exception as e:
        return jsonify({'error': str(e)})    
if __name__ == '__main__':
    web.run(host='0.0.0.0',port=5000,debug=True)

#now we make the html front end page for prediction    
        