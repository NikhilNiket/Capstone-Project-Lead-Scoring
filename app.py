import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,send_from_directory
from flask_cors import CORS, cross_origin
import pickle

#instatiating a flask app
app = Flask(__name__,static_folder='templates/public')

#resolving cors issues
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#encoder pickle file
with open("encoder_test_lead_scoring.pkl" , 'rb') as file:  
    enc = pickle.load(file)

#loading the model
with open("final_test_lead_scoring.pkl", 'rb') as file:  
    Pickled_ada_Model = pickle.load(file)

@app.route('/')
def home():
    #serving html file
    return render_template('index.html')

@app.route('/public/<path:name>')
def show_css(name):
    #serving static files
    return send_from_directory(app.static_folder,name)

@cross_origin()


@app.route('/predict',methods=['POST'])
def predict():
    #parsing body to dict
    resp = request.get_json(force=True)
    pred_df = pd.DataFrame(resp,index=[0])
    pred = enc.transform(pred_df).toarray()
    a,b = str(Pickled_ada_Model.predict(pred)[0]),int(Pickled_ada_Model.predict_proba(pred)[0][1]*100)
    print(a,b)
    #predicting the output from the html page served data
    return jsonify(conversion_status=a,score=b)
    #return str(Pickled_ada_Model.predict(pred)[0])
    #print(str(Pickled_ada_Model.predict(pred)[0]))
    #print(int(Pickled_ada_Model.predict_proba(pred)[0][1]*100))



if __name__ == "__main__":
    app.run(debug=True)
