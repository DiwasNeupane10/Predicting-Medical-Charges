from flask import Flask,render_template,request,jsonify,redirect
import joblib
import numpy as np
import pandas as pd
app=Flask(__name__)


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    if request.method=='POST':
        features=get_features(request.form.items())#get features from the request
        input_df=pd.DataFrame([features])#convert into dataframe
        mapped_input_df=map_binary_category(input_df)  
        final_input_df=process_input_df(mapped_input_df)
        df_html=final_input_df.to_html(classes='table table-striped',index=False)
        model=joblib.load('./medical_charge_predict_model.sav')
        prediction=model.predict(final_input_df)[0]
        
        return render_template('prediction.html',table=df_html,predict=prediction)
    else: 
        return redirect('/')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Invalid input, JSON expected"}), 400
    data = request.get_json()
    # Define expected data types
    expected_types = {
        "age": int,
        "bmi": float,
        "children": int,
        "smoker": str,
        "sex": str,
        "region": str
    }
    expected_str_input={
        "smoker":['Yes','No'],
        "sex":['Male','Female'],
        "region":['northeast','northwest','southwest','southeast']
    }
    
    # Validate the JSON data
    errors = validate_json(data, expected_types,expected_str_input)
    if errors:
        return jsonify({"errors": errors}), 400
    input_df=pd.DataFrame([data])
    mapped_input_df=map_binary_category(input_df)  
    final_input_df=process_input_df(mapped_input_df)
    model=joblib.load('./medical_charge_predict_model.sav')
    prediction=model.predict(final_input_df)[0]

    return jsonify({"Predicted Medical Charge":f"${prediction}"})



def get_features(x):
    features={}
    for key ,value in x:
        if key in['age','children']:
            features[key]=int(value)
        elif key=='bmi':
            features[key]=float(value)
        elif key in ['region','sex','smoker']:
            features[key]=value
        else:
            continue
    return features

def map_binary_category(input_df):
    sex_map={'Male':1,'Female':0}
    smoker_map={'Yes':1,'No':0}
    if 'sex' in input_df.columns:
        input_df['sex']=input_df['sex'].map(sex_map)
    if 'smoker' in input_df.columns:
        input_df['smoker']=input_df['smoker'].map(smoker_map)
    return input_df

def process_input_df(input_df):
    numeric_cols=['age','bmi','children']
    scaler=joblib.load('./scaler.sav')
    input_df[numeric_cols]=scaler.transform(input_df[numeric_cols])
    categorical_cols=['region']
    encoder=joblib.load('./encoder.sav')
    encoded_categories=list(encoder.get_feature_names_out(categorical_cols))
    onehot=encoder.transform(input_df[categorical_cols]).toarray()
    input_df[encoded_categories]=onehot
    input_df=input_df.drop(columns=['region'],axis=1)
    return input_df


def validate_json(data, expected_types,expected_str_input):
    errors = []
    for key, expected_type in expected_types.items():
        if key not in data:
            errors.append(f"Missing field: {key}")
        elif not isinstance(data[key], expected_type):
            errors.append(f"Incorrect type for field '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
    for key,value in expected_str_input.items():
        if data[key] not in value:
            errors.append(f"Incorrect value for the field '{key}' expected '{expected_str_input[key]}'")
    return errors



if __name__ == '__main__':
    app.run(debug=True)