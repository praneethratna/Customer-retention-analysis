import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
col2 = ['MultipleLines','InternetService', 'Contract']
c0 = ['gender']
c1 = [ 'SeniorCitizen', 'Partner', 'Dependents']
c2 = ['tenure']
c3 = ['PhoneService', 'PaperlessBilling']
c4 = ['MonthlyCharges']
col1 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
'StreamingTV', 'StreamingMovies']
#Create Flask app
app = Flask(__name__)

#Load the pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    pred_text = ""
    return render_template('index.html', prediction_text = pred_text)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    #convert data into a dataframe
    value = dict()
    data = request.form.to_dict()
    for col in c0:
        if data[col] == 'Male':
            value[col] = 1
        else:
            value[col] = 0

    for col in c1:
       if data[col] == 'Yes':
        value[col] = 1
       else:
        value[col] = 0
    value[c2[0]] = data[c2[0]]
    for col in c3:
        if data[col] == 'Yes':
          value[col] = 1
        else:
          value[col] = 0

    value[c4[0]] = data[c4[0]]

    if data['MultipleLines'] == 'No':
        value['MultipleLines_'+'No phone service'] = 0
        value['MultipleLines_'+'Yes'] = 0
    elif data['MultipleLines'] == 'No phone service':
        value['MultipleLines_'+'No phone service'] = 1
        value['MultipleLines_'+'Yes'] = 0
    else:
        value['MultipleLines_'+'No phone service'] = 0
        value['MultipleLines_'+'Yes'] = 1

    if data['InternetService'] == 'No':
        value['InternetService_'+'Fiber optic'] = 0
        value['InternetService_'+'No'] = 1
    elif data['InternetService'] == 'Fiber optic':
        value['InternetService_'+'Fiber optic'] = 1
        value['InternetService_'+'No'] = 0
    else:
        value['InternetService_'+'Fiber optic'] = 0
        value['InternetService_'+'No'] = 0


    for col in col1:
        if data[col] == 'No':
            value[col + "_" + "No internet service"] = 0
            value[col + "_" + "Yes"] = 0
        if data[col] == 'No internet service':
            value[col + "_" + "No internet service"] = 1
            value[col + "_" + "Yes"] = 0
        if data[col] == 'Yes':
            value[col + "_" + "No internet service"] = 0
            value[col + "_" + "Yes"] = 1

    if data['Contract'] == 'Month-to-month':
        value['Contract_'+'One year'] = 0
        value['Contract_'+'Two year'] = 0
    elif data['Contract'] == 'One year':
        value['Contract_'+'One year'] = 1
        value['Contract_'+'Two year'] = 0
    else:
        value['Contract_'+'One year'] = 0
        value['Contract_'+'Two year'] = 1

    if data['PaymentMethod'] == 'Bank transfer (automatic)':
        value['PaymentMethod_'+'Credit card (automatic)'] = 0
        value['PaymentMethod_'+'Electronic check'] = 0
        value['PaymentMethod_'+'Mailed check'] = 0
    elif data['PaymentMethod'] == 'Credit card (automatic)':
        value['PaymentMethod_'+'Credit card (automatic)'] = 1
        value['PaymentMethod_'+'Electronic check'] = 0
        value['PaymentMethod_'+'Mailed check'] = 0
    elif data['PaymentMethod'] == 'Electronic check':
        value['PaymentMethod_'+'Credit card (automatic)'] = 0
        value['PaymentMethod_'+'Electronic check'] = 1
        value['PaymentMethod_'+'Mailed check'] = 0
    else:
        value['PaymentMethod_'+'Credit card (automatic)'] = 0
        value['PaymentMethod_'+'Electronic check'] = 0
        value['PaymentMethod_'+'Mailed check'] = 1
    df = pd.DataFrame(value, index = [0])
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    print(df['MonthlyCharges'])

    # Make prediction using model loaded from disk as 'model'.
    prediction = model.predict(df[0:1])
    if prediction[0] == 0:
        pred_text = 'The customer has not left the company within the last month'
    else:
        pred_text = 'The customer has left the company within the last month'
    return render_template('index.html', prediction_text='{}'.format(pred_text))

if __name__ == '__main__':
    app.run(debug=True)