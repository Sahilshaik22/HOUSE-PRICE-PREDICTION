import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def Build_pipeline(num_attributes,cat_attributes):
    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("StandardScaler",StandardScaler())
    ]) 
    cat_pipeline = Pipeline([
        ("Onehotecoder",OneHotEncoder(handle_unknown="ignore"))
    ])
    Full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attributes),
        ("cat",cat_pipeline,cat_attributes)
    ])
    
    return Full_pipeline


if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")
    housing["income_category"] = pd.cut(housing["median_income"],bins = [0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size = 0.2, random_state = 42)

    for train_data, test_data in split.split(housing,housing["income_category"]):
        stratified_train_data = housing.iloc[train_data].drop("income_category",axis=1)
        housing.iloc[test_data].drop("income_category",axis=1).to_csv("input.csv",index=False)

    housing = stratified_train_data.copy()
    
    housing_features = housing.drop("median_house_value",axis = 1)
    housing_lables = housing["median_house_value"].copy()

    num_attributes = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]
    
    pipeline= Build_pipeline(num_attributes,cat_attributes)
    Features = pipeline.fit_transform(housing_features)
    
    model= RandomForestRegressor(random_state=42)
    model.fit(Features,housing_lables)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    
    print("Your Model is Trained successfully Congrats")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions
    input_data.to_csv("output.csv",index=False)
    print("Interence is completed your predictions are waiting for You Enjoy")
