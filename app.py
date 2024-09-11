from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd
app=Flask(__name__)


@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features={}
    for key ,value in request.form.items():
        if key in['age','children']:
            features[key]=int(value)
        elif key=='bmi':
            features[key]=float(value)
        elif key in ['region','sex','smoker']:
            features[key]=value
        else:
            continue
        
    sex_map={'Male':1,'Female':0}
    smoker_map={'Yes':1,'No':0}
    input_df=pd.DataFrame([features])
    if 'sex' in input_df.columns:
        input_df['sex']=input_df['sex'].map(sex_map)
    if 'smoker' in input_df.columns:
        input_df['smoker']=input_df['smoker'].map(smoker_map)
    numeric_cols=['age','bmi','children']
    scaler=joblib.load('./scaler.sav')
    input_df[numeric_cols]=scaler.transform(input_df[numeric_cols])
    categorical_cols=['region']
    encoder=joblib.load('./encoder.sav')
    encoded_categories=list(encoder.get_feature_names_out(categorical_cols))
    onehot=encoder.transform(input_df[categorical_cols]).toarray()
    input_df[encoded_categories]=onehot
    input_df=input_df.drop(columns=['region'],axis=1)
    df_html = input_df.to_html(classes='table table-striped', index=False)

    model=joblib.load('./medical_charge_predict_model.sav')
    prediction=model.predict(input_df)[0]
    
    return render_template('index.html',predict=prediction)
    

if __name__ == '__main__':
    app.run(debug=True)