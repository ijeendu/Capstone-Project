##writefile score.py
import pickle
import json
import numpy 
from sklearn import joblib
from sklearn.tree import DecisionTreeClassifier
from azureml.core.model import Model
import time
import os

def init():
    global model
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))

    # note here "AZUREML_MODEL_DIR" is the name of the model registered under the workspace
    # this call should return the path to the model.pkl file on the local disk.
    #model_path = Model.get_model_path(model_name = 'best-model')	
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sklearn_minst_model.pkl') 
	
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


# Pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
		#make prediction
        result = model.predict(data)
        # Log the input and output data to appinsights:
        info = {
            "input": raw_data,
            "output": result.tolist()
            }
        print(json.dumps(info))
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error