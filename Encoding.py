import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



# read csv
df = pd.read_csv("BankChurners.csv")


# seprately encode ordinal variables(Income_Category, Card_Category, Attrition_Flag, Education_Level)
Income_Category_map = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
                       '$80K - $120K': 3, '$120K +': 4, 'Unknown': 5}


Card_Category_map = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}


Attrition_Flag_map = {'Existing Customer': 0, 'Attrited Customer': 1}

Education_Level_map = {'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3,
                       'Post-Graduate': 4, 'Doctorate': 5, 'Unknown': 6}



df.loc[:, 'Income_Category'] = df['Income_Category'].map(Income_Category_map)
df.loc[:, 'Card_Category'] = df['Card_Category'].map(Card_Category_map)
df.loc[:, 'Attrition_Flag'] = df['Attrition_Flag'].map(Attrition_Flag_map)
df.loc[:, 'Education_Level'] = df['Education_Level'].map(Education_Level_map)


# LabelEncoder
labelencoder = LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df.loc[:, i] = labelencoder.fit_transform(df.loc[:, i])
