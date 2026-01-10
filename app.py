from flask import Flask, render_template_string, render_template, request
import pickle
import numpy as np
import math, time
import pandas as pd
from notebooks.fuzzy_logic import Fuzzy
#import sklearn


app = Flask(__name__)


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
    with open(r"explainers\shap\Cost_Regression_Explainer.pkl", 'rb') as f:
        cost_shap = pickle.load(f)
    with open(r"explainers\lime\Cost_Regression_Explainer.pkl", 'rb') as f:
        cost_lime = pickle.load(f)
    #time explainers
    with open(r"explainers\shap\Time_Regression_Explainer.pkl", 'rb') as f:
        time_shap = pickle.load(f)
    with open(r"explainers\lime\Time_Regression_Explainer.pkl", 'rb') as f:
        time_lime = pickle.load(f)
    #session explainers
    with open(r"explainers\shap\Long_Session_Explainer.pkl", 'rb') as f:
        session_shap = pickle.load(f)
    with open(r"explainers\lime\Long_Session_Explainer.pkl", 'rb') as f:
        session_lime = pickle.load(f)
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

def prepare_inputs(form_data, model_type=None):
    """
    Convert the form input into the data shape our model expect
    """
    
    energy = float(form_data[""])

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
    
    
    
    ######## Long session feature history to be used in the admin panel
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

    cost = cost_model.predict(cost_df)
    time = time_model.predict(time_df)
    
    urgency = int(request.form.get('urgency', 5))
    budget = int(request.form.get('budget', 5))
    
    fz = Fuzzy(urgency, budget, cost, time)
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

    prediction = True
    return render_template("car.html", prediction=prediction, cost=cost, time=time, comfort_score = int(mom), advice_text=advice)
@app.route('/session')
def admin_dashboard():
    df_history=pd.DataFrame(long_session_dict)
    
    predicted = False
    #session_data = []
    anomaly_list = []
    
    if not df_history.empty:
        predicted = True
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
        #convert df to list of dictionaries
        session_data = df_history.to_dict(orient="records")
        anomaly_list = [i for i in session_data]
    else:
        session_data = []
    return render_template('admin.html', prediction=predicted, sessions=session_data, anomalies=anomaly_list)

if __name__ == "__main__":
    app.run(debug=True)
