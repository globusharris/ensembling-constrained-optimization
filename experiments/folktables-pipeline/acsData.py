# Wrapper to load in the ACS data
#import json
from folktables import ACSDataSource, adult_filter
#from sklearn.model_selection import train_test_split

def get_data(acs_year, acs_states, acs_horizon='1-Year', acs_survey='person'):
    
    # acs_year: 2014-2018
    # acs_states: any list of state abbreviations e.g. ['NY']
    # acs_horizon: '1-Year' or '5-Year'
    # acs_survey: 'person' or 'household'
    
    target_columns = ["AGEP","COW","SCHL","MAR","OCCP","POBP","RELP","WKHP","SEX","RAC1P","ST"]
    label_column = "PINCP"

    # Pull in raw data
    data_source = ACSDataSource(survey_year=acs_year, horizon=acs_horizon, survey=acs_survey)
    acs_data = data_source.get_data(states=acs_states, download=True)
    # Filtering to mimic filtering in Adult
    filtered_data = adult_filter(acs_data) #mimics filtering used in Adult dataset
    # Pulling out features and targets. This data has no nans so don't need to worry about that. 
    features = filtered_data[target_columns]
    target = filtered_data[label_column]
    
    return features, target

