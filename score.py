##writefile score.py
import pickle
import json
import numpy, pandas
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from azureml.core.model import Model
import time
import os

def init():
    global model
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_automl_model.pkl') 
    print('Found Model: ', os.path.isfile(model_path))
	
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


# can pass in multiple rows for scoring
def run(raw_data):
    try:
        # load raw data
        data = json.loads(raw_data)['data']
        # convert to input format - DataFrame
        data = pandas.DataFrame.from_dict(data)             
        #make prediction
        result = model.predict(data)
        # Log the input and output data to appinsights:
        info = {
            "input": raw_data,
            "output": result.tolist()
            }
        print(json.dumps(info))
        # return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error