import pickle
import pandas as pd


with open("encoder_test_lead_scoring.pkl" , 'rb') as file:  
    enc = pickle.load(file)

data_1={
    "total_visits":[2],
    "total_time_spent_on_website":[1532],
    "lead_origin":["Landing Page Submission"],
    "lead_source":["Direct Traffic"],
    "specialization":["Business Administration"],
    "how_did_you_hear_about_x_education":["Online Search"],
    "what_is_your_current_occupation":["Student"],
    "tags":["Will revert after reading the email"],
    "lead_quality":["Might be"],
    "lead_profile":["Potential Lead"],
    "last_notable_activity":["Email Opened"]
}

pred_df = pd.DataFrame(data_1)
pred = enc.transform(pred_df).toarray()

with open("../final_test_lead_scoring.pkl", 'rb') as file:  
    Pickled_ada_Model = pickle.load(file)


print('Predicted values:',Pickled_ada_Model.predict(pred))