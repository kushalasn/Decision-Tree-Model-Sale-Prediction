import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
from flask_cors import CORS

app=Flask(__name__)
CORS(app)


model=pickle.load(open('model.pkl','rb'))



@app.route('/',methods=['GET'])
def webprint():
    return (render_template('index.html'))

@app.route('/predict',methods=['POST'])
def predict():

    print(request.form)

    occupation=request.form['occupation']
    print(occupation)

    income_group=request.form['INCOME_GROUP']
    customer_since=request.form['CUSTOMER_SINCE']
        
    loyalty_program=request.form['LOYALTY_PROGRAM']
    past_purchase_normalised=request.form['PAST_PURCHASE_Normalised']
    age_normalised=request.form['AGE_Normalised']
       
         
    input_variables = pd.DataFrame([[occupation ,income_group,customer_since,loyalty_program, past_purchase_normalised,age_normalised]],
                                       columns=['OCCUPATION','INCOME_GROUP','CUSTOMER_SINCE','LOYALTY_PROGRAM','PAST_PURCHASE_Normalised','AGE_Normalised'],
                                       dtype=float)

    prediction= model.predict(input_variables)
    print(prediction)
    if prediction==1:
        p='YES'
    else:
        p='NO'

            
    result={"occupation":occupation,"INCOME_GROUP":income_group,"CUSTOMER_SINCE":customer_since,"LOYALTY_PROGRAM":loyalty_program,"PAST_PURCHASE_Normalised":past_purchase_normalised,"AGE_Normalised":age_normalised,"Will the Person Purchase":p}
    result = jsonify(result)
    print(result)
    result.headers.add('Access-Control-Allow-Origin', '*')
    #return result
    return render_template('index.html',result=result)



if __name__=='__main__':
    app.run(debug=True,port=8088)    




  
