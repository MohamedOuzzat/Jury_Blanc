import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif



def EDA():

    df = pd.read_csv("/opt/airflow/data/student_mental_health.csv")

    num_cols = [
        'Age',
        'Year_of_Study',
        'Sleep_Hours',
        'Study_Hours_Per_Week',
        'GPA',
        'Social_Media_Hours',
        'Screen_Time_Hours',
        'Financial_Stress',
        'Family_Support',
        'Attendance_Rate'
    ]

    nominal_cols = [
        'Gender',
        'Part_Time_Job',
        'Previous_Mental_Health_History'
    ]

    ordinal_cols = [
        'Physical_Activity',
        'Diet_Quality'
    ]

    target_col = "Mental_Health_Risk"
    
    print("EDA completed successfully")

    return {
        "num_cols": num_cols,
        "nominal_cols": nominal_cols,
        "ordinal_cols": ordinal_cols,
        "target_col": target_col
    }






     