from flask import Flask, render_template_string, render_template, request
import pickle
import numpy as np
import math, time
import pandas as pd
from notebooks.fuzzy_logic import Fuzzy
from notebooks.XAIGen import XAIHandler
import matplotlib
matplotlib.use('Agg')  # <--- Add this line BEFORE importing pyplot
import matplotlib.pyplot as plt
import shap
import joblib
import dill
from notebooks.RecomEngine import ActionRecommender
#import sklearn


app = Flask(__name__)

recommender = ActionRecommender(data_path=r"data\preprocessed_encoded_data.pkl")
#recommender.preprocess_Train() #train the rl model.
recommender.load_model(r"..\saved_models\rl_model.pkl")
handler = XAIHandler()

try:
    with open(r"saved_models\cost_regression.pkl", 'rb') as f:
        cost_model = pickle.load(f)
    with open(r"saved_models\time_regression.pkl", 'rb') as f:
        time_model = pickle.load(f)
    with open(r"saved_models\long_session.pkl", 'rb') as f:
        session_model = pickle.load(f)
except Exception as e:
    print(f"There was an Exception {e} loading main models")

try:
    #Cost explainers
    cost_shap = joblib.load(r"explainers\shap\Cost_Regression_Explainer.pkl")
    with open(r"explainers\lime\Cost_Regression_Explainer.pkl", 'rb') as f:
        cost_lime = dill.load(f)

    #time explainers
    time_shap = joblib.load(r"explainers\shap\Time_Regression_Explainer.pkl")
    with open(r"explainers\lime\Time_Regression_Explainer.pkl", 'rb') as f:
        time_lime = dill.load(f)
    #session explainers
    session_shap = joblib.load(r"explainers\shap\Long_Session_Explainer.pkl")
    with open(r"explainers\lime\Long_Session_Explainer.pkl", 'rb') as f:
        session_lime = dill.load(f)
except Exception as e:
    print(f"There was an Exception {e} loading explainers")
#for the anomaly model
num_col = ["Energy Consumed (kWh)", "Charging Rate (kW)", "Day of Week", "Battery Capacity (kWh)", "State of Charge (Start %)", "Distance Driven (since last charge) (km)"]
cat_col =["User Type", "Charger Type", "Charging Station Location_Chicago", "Charging Station Location_Houston", "Charging Station Location_Los Angeles", "Charging Station Location_New York", "Charging Station Location_San Francisco"]

try:
    with open(r"saved_models/One_CLass_SVM.pkl", "rb") as f:
        ocsvm = pickle.load(f)
except Exception as e:
    print(f"There was an Exception {e} loading anomaly models")



@app.route('/')
def home():
    return render_template('index.html')
@app.route("/car")
def car_dashboard():
    return render_template('car.html')

long_session_dict={
    #ocsvm columns
    "Energy Consumed (kWh)": [],
    "Charging Rate (kW)": [],
    #rest of columns that are long session or shared.
    "Battery Capacity (kWh)" : [],
    "Time of Day" : [],
    "Day of Week": [],
    "State of Charge (Start %)": [],
    "Distance Driven (since last charge) (km)": [],
    "Temperature (°C)" : [],
    "Vehicle Age (years)" : [],
    "Charger Type" : [],
    "User Type" : [],
    "Vehicle Model_BMW i3" : [],
    "Vehicle Model_Chevy Bolt" : [],
    "Vehicle Model_Hyundai Kona" : [],
    "Vehicle Model_Nissan Leaf" : [],
    "Vehicle Model_Tesla Model 3" : [],
    "Charging Station Location_Chicago" : [],
    "Charging Station Location_Houston" : [],
    "Charging Station Location_Los Angeles" : [],
    "Charging Station Location_New York" : [],
    "Charging Station Location_San Francisco" : [],
}
df = joblib.load("data\preprocessed_encoded_data.pkl") #for the sake of populating the session page with data
fz = Fuzzy(df)
subset = df.head(5)
for k, v in long_session_dict.items():
    if k in subset.columns:
        long_session_dict[k] = subset[k].tolist()
    else:
        pass #maybe add functions here to populate the list
@app.route("/predict", methods=["POST"])
def predict():
    prediction = False
    #ordinal features
    day_of_week = int(request.form['day_week'])
    time_of_day = int(request.form['time_day'])
    charger_type = int(request.form['charger_type'])
    user_type = int(request.form['user_type'])

    #nominal features
    model = "Vehicle Model_" + request.form["vehicle"]
    charger_location = "Charging Station Location_" + request.form["location"]
    #numerical features
    energy = float(request.form['energy'])
    charge_rate = float(request.form['charge_rate'])
    battery_cap = float(request.form['battery_cap'])
    start_soc = float(request.form['start_soc'])
    dist_driven = float(request.form['dist_driven'])
    temp = float(request.form['temp'])
    vehicle_age = int(request.form['age'])


    cost_regression_dict = {
        "Energy Consumed (kWh)": energy,
        "Charging Rate (kW)": charge_rate,
        "Time of Day": time_of_day,
        "Day of Week": day_of_week,
        "Charger Type": charger_type,
        "User Type": user_type,
        "Vehicle Model_BMW i3" : 0,
        "Vehicle Model_Chevy Bolt" : 0,
        "Vehicle Model_Hyundai Kona": 0,
        "Vehicle Model_Nissan Leaf" : 0,
        "Vehicle Model_Tesla Model 3" : 0,
        "Charging Station Location_Chicago" : 0,
        "Charging Station Location_Houston" : 0,
        "Charging Station Location_Los Angeles" : 0,
        "Charging Station Location_New York" :0,
        "Charging Station Location_San Francisco" :0,
    }
    time_regression_dict = {
        "Battery Capacity (kWh)": battery_cap,
        "Charging Rate (kW)": charge_rate,
        "State of Charge (Start %)": start_soc,
        "Distance Driven (since last charge) (km)": dist_driven,
        "Temperature (°C)": temp,
        "Vehicle Age (years)" : vehicle_age,
        "Charger Type": charger_type,
        "User Type": user_type,
        "Vehicle Model_BMW i3" : 0,
        "Vehicle Model_Chevy Bolt" : 0,
        "Vehicle Model_Hyundai Kona": 0,
        "Vehicle Model_Nissan Leaf" : 0,
        "Vehicle Model_Tesla Model 3" : 0,
        "Charging Station Location_Chicago" : 0,
        "Charging Station Location_Houston" : 0,
        "Charging Station Location_Los Angeles" : 0,
        "Charging Station Location_New York" :0,
        "Charging Station Location_San Francisco" :0,
    }
    cost_df = pd.DataFrame([cost_regression_dict])
    time_df = pd.DataFrame([time_regression_dict])
    if model in cost_df and model in time_df:
        cost_df[model] = 1
        time_df[model] = 1
    if charger_location in cost_df and charger_location in time_df:
        cost_df[charger_location] = 1
        time_df[charger_location] = 1
    
    
    
    ######## Long session feature history to be used in the admin panel######################
    long_session_dict["Battery Capacity (kWh)"].append(battery_cap)
    long_session_dict["Time of Day"].append(time_of_day)
    long_session_dict["Day of Week"].append(day_of_week)
    long_session_dict["State of Charge (Start %)"].append(start_soc)
    long_session_dict["Distance Driven (since last charge) (km)"].append(dist_driven)
    long_session_dict["Temperature (°C)"].append(temp)
    long_session_dict["Vehicle Age (years)"].append(vehicle_age)
    long_session_dict["Charger Type"].append(charger_type)
    long_session_dict["User Type"].append(user_type)
    long_session_dict["Energy Consumed (kWh)"].append(energy)
    long_session_dict["Charging Rate (kW)"].append(charge_rate)
    cat_keys = [
        "Vehicle Model_BMW i3", "Vehicle Model_Chevy Bolt", "Vehicle Model_Hyundai Kona",
        "Vehicle Model_Nissan Leaf", "Vehicle Model_Tesla Model 3",
        "Charging Station Location_Chicago", "Charging Station Location_Houston",
        "Charging Station Location_Los Angeles", "Charging Station Location_New York",
        "Charging Station Location_San Francisco"
    ]

    for key in cat_keys:
        if key == model or key == charger_location:
            long_session_dict[key].append(1)
        else:
            long_session_dict[key].append(0)
    #############PREDICTION#################
    cost = cost_model.predict(cost_df)
    time = time_model.predict(time_df)
    #######XAI LIME######
    exp_cost_lime = handler.explain_lime(
        cost_df,
        model=cost_model, 
        FEATURE_NAMES=cost_df.columns.tolist(),
        explainer=cost_lime,
        mode="regression"
    )
    exp_time_lime = handler.explain_lime(
        time_df,
        model=time_model,
        FEATURE_NAMES=time_df.columns.tolist(),
        explainer=time_lime,
        mode="regression"
    )
    #######XAI SHAP########
    exp_cost_shap = handler.explain_shap(
        cost_df,
        model=cost_model, 
        FEATURE_NAMES=cost_df.columns.tolist(),
        explainer=cost_shap,
        mode="regression"
    )
    exp_time_shap = handler.explain_shap(
        time_df,
        model=time_model,
        FEATURE_NAMES=time_df.columns.tolist(),
        explainer=time_shap,
        mode="regression"
    )
    ##########FUZZY LOGIC##############
    urgency = int(request.form.get('urgency', 5))
    budget = int(request.form.get('budget', 5))
    fz.Fuzz(urgency, budget, cost, time)
    mom = fz.defuzz()
    advice = None
    if mom is None:
        advice = "No decision possible"
    elif mom < 30:
        advice = "You are uncomfortable"
    elif mom > 70:
        advice = "Do anything"
    else:
        advice = "idk man do whatever u want"
    
    
    ########RECOMMENDATION AGENT##########
    location_mapping = {
            "Chicago": 0,
            "Houston": 1,
            "Los Angeles": 2,
            "New York": 3,
            "San Francisco": 4
    }
    location = request.form["location"]
    location_idx = location_mapping[location]
    recommendation = recommender.recommend_action(
        battery_capacity=battery_cap,
        location_idx=location_idx,
        charger_type=charger_type,
        time_of_day=time_of_day)
    prediction = True
    return render_template(
        "car.html", 
        prediction=prediction, 
        cost=cost, 
        time=time, 
        comfort_score = 
        int(mom), 
        advice_text=advice, 
        recommendation_text=recommendation,
        cost_lime=exp_cost_lime['explanation'],
        cost_shap=exp_cost_shap['explanation'],
        time_lime=exp_time_lime['explanation'],
        time_shap=exp_time_shap['explanation']
    )

#Fuzzy Clustering
u_matrix, cntr, descriptors = fz.cluster(4, 2.0, 0.005, 1000,["Energy Consumed (kWh)","Charging Rate (kW)", "Temperature (°C)"])

@app.route('/session')
def admin_dashboard():
    df_history=pd.DataFrame(long_session_dict)
    do_once = 0
    predicted = False
    #session_data = []
    anomaly_list = {}
    results = []
    if not df_history.empty:
        predicted = True
        df_history = df_history.apply(pd.to_numeric, errors='ignore')
        long_session_cols = [
            "Battery Capacity (kWh)", "Time of Day", "Day of Week", 
            "State of Charge (Start %)", "Distance Driven (since last charge) (km)", 
            "Temperature (°C)", "Vehicle Age (years)", "Charger Type", "User Type",
            "Vehicle Model_BMW i3", "Vehicle Model_Chevy Bolt", 
            "Vehicle Model_Hyundai Kona", "Vehicle Model_Nissan Leaf", 
            "Vehicle Model_Tesla Model 3", "Charging Station Location_Chicago", 
            "Charging Station Location_Houston", "Charging Station Location_Los Angeles", 
            "Charging Station Location_New York", "Charging Station Location_San Francisco"
        ]
        prediction = session_model.predict(df_history[long_session_cols])
        df_history["Is_Long"] = prediction
        
        #calculate anomaly
        score_ocsvm = -ocsvm.decision_function(df_history[num_col+cat_col])
        anomaly_ocsvm = (ocsvm.predict(df_history[num_col+cat_col]) == -1).astype(int)
        
        df_history['anomaly_score'] = score_ocsvm.round(4)
        df_history['is_anomaly'] = anomaly_ocsvm
        anomaly_list = df_history[["anomaly_score", "is_anomaly"]].to_dict(orient="records")
        #convert df to list of dictionaries
        session_data = df_history.to_dict(orient="records")


        for row in range(len(df_history)):
            ###XAI
            ##LIME
            exp_cost_lime = handler.explain_lime(
                df_history[long_session_cols].iloc[row],
                model=session_model,
                FEATURE_NAMES=long_session_cols,
                explainer=session_lime,
                mode="classification"
            )
            session_data[row]['lime'] = exp_cost_lime['explanation']
            ##shap
            exp_time_shap = handler.explain_shap(
                df_history[long_session_cols].iloc[row],
                model=session_model,
                FEATURE_NAMES=long_session_cols,
                explainer=session_shap,
                mode="classification"
            )
            session_data[row]['shap'] = exp_time_shap['explanation']
            ####CLUSTERING
            row_data = df_history.iloc[row]
            row_data = row_data[["Energy Consumed (kWh)","Charging Rate (kW)", "Temperature (°C)"]]
            dominant_cluster, memberships = fz.predict_cluster(
                input=row_data,
                features=["Energy Consumed (kWh)","Charging Rate (kW)", "Temperature (°C)"],
                centers=cntr
            )
            best_cluster = descriptors[dominant_cluster]
            row_results = {
                "id": row + 1,
                "memberships": memberships,
                "best_cluster": best_cluster,
                "cluster_name": best_cluster
            }
            results.append(row_results)
    else:
        session_data = []
    return render_template('admin.html', prediction=predicted, sessions=session_data, anomalies=anomaly_list, analysis_results=results,cluster_descriptors=descriptors)

if __name__ == "__main__":
    app.run(debug=True)
