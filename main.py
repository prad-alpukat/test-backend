from typing import Union
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
import numpy as np

def custom_label_encode(arr):
    unique_vals = np.unique(arr)
    encode_dict = {val: idx for idx, val in enumerate(unique_vals)}
    return np.vectorize(encode_dict.get)(arr), encode_dict

def custom_standard_scale(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std if std != 0 else arr

def predict_loan_approval(model, age, balance, day, duration, campaign, pdays, previous, 
                          job, marital, education, default, housing, loan, contact, month, poutcome):
    # Create a dictionary to store the data
    data_dict = {
        'age': np.array([age]),
        'balance': np.array([balance]),
        'day': np.array([day]),
        'duration': np.array([duration]),
        'campaign': np.array([campaign]),
        'pdays': np.array([pdays]),
        'previous': np.array([previous]),
        'job': np.array([job]),
        'marital': np.array([marital]),
        'education': np.array([education]),
        'default': np.array([default]),
        'housing': np.array([housing]),
        'loan': np.array([loan]),
        'contact': np.array([contact]),
        'month': np.array([month]),
        'poutcome': np.array([poutcome])
    }
    
    # List of numerical columns to scale
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    # Custom encode categorical features
    encode_dicts = {}
    for feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        data_dict[feature], encode_dicts[feature] = custom_label_encode(data_dict[feature])
    
    # Custom scale numerical features
    for feature in numerical_columns:
        data_dict[feature] = custom_standard_scale(data_dict[feature])

    # Combine all features into a single NumPy array
    data_array = np.column_stack([data_dict[feature] for feature in data_dict.keys()])

    # Predict the loan approval
    prediction = model.predict(data_array)
    return prediction[0].tolist()
    
joblib_file = "./model.pkl"
model = joblib.load(joblib_file)

@app.get("/")
async def read_root():
   result = predict_loan_approval( model,58,2143,5,261,1,-1,0,'management','married','tertiary','no','yes','no','unknown','may','unknown')
   return {"message": "Welcome to the API", "result": result}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}