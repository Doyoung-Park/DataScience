import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing



# read csv
df = pd.read_csv("BankChurners.csv")



# Feature Selection
new_df_columns = ['Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Revolving_Bal', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count',
          'Total_Amt_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Contacts_Count_12_mon', 'Months_Inactive_12_mon', 'Credit_Limit', 'Attrition_Flag']
new_df = pd.DataFrame(df, columns=new_df_columns)



# Scaling
features_scaling = ['Total_Ct_Chng_Q4_Q1', 'Total_Amt_Chng_Q4_Q1',
                    'Credit_Limit', 'Avg_Utilization_Ratio']
df_features_scaling = pd.DataFrame(new_df, columns=features_scaling)
df_remaining_features = new_df.drop(columns=features_scaling)


def scaling(feature, scale):
    if scale == 'StandardScaling':
        scaler = preprocessing.StandardScaler()
        scaled_feature = scaler.fit_transform(feature)
    elif scale == 'RobustScaling':
        scaler = preprocessing.RobustScaler()
        scaled_feature = scaler.fit_transform(feature)
    elif scale == 'MinMaxScaling':
        scaler = preprocessing.MinMaxScaler()
        scaled_feature = scaler.fit_transform(feature)
    else:
        scaler = preprocessing.MaxAbsScaler()
        scaled_feature = scaler.fit_transform(feature)
        
    df_scaled_features = pd.DataFrame(scaled_feature, columns=features_scaling, index=df_remaining_features.index)
    return df_scaled_features




# 1) Standard Scaling
df_scaled_features = scaling(df_features_scaling,'StandardScaling')
df_standard_scaled_data = pd.concat([df_remaining_features, df_scaled_features], axis=1)   


# 2) Robust Scaling
df_scaled_features = scaling(df_features_scaling,'RobustScaling')
df_robust_scaled_data = pd.concat([df_remaining_features, df_scaled_features], axis=1)            


# 3) MinMax Scaling
df_scaled_features = scaling(df_features_scaling,'MinMaxScaling')     
df_minmax_scaled_data = pd.concat([df_remaining_features, df_scaled_features], axis=1)                                  


# 4) MaxAbs Scaling
df_scaled_features = scaling(df_features_scaling,'MaxAbsScaling')
df_maxabs_scaled_data = pd.concat([df_remaining_features, df_scaled_features], axis=1)