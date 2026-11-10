import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif

def EDA():

    df=pd.read_csv("../data/student_mental_health.csv")
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

    target_col="Mental_Health_Risk"

    # Remove the target column from X
    X = df[num_cols].drop(columns=[target_col], errors="ignore")

    # 2. Encode the target variable
    y = df[target_col]

    print("X : ",X)
    print("Y : ",y)
    print("numerique : ",X)
    print("X : ",X)

    return X,y,num_cols,nominal_cols,ordinal_cols,df





        # Dimensions
    # print("\nNombre de lignes :", df.shape[0])
    # print("Nombre de colonnes :", df.shape[1])

    # # Informations generales
    # print("\nInformations generales : ",df.info())

    # # Types de donnees
    # print("\nTypes de donnees : ",df.dtypes)


    # print("\nNombre de doublons :", df.duplicated().sum())
    # print("Nombre de manquantes :", df.isnull().sum()) 

    

    # # Valeurs uniques des variables categorielles
    # print("\nValeurs uniques des variables categorielles")
    # categorial=df.select_dtypes(include="object").columns
    # for col in categorial:
    #     print(f"\n{col}:")
    #     print(df[col].value_counts())


    # #Valuers aberrantes numerique
    # print("\nValuers aberrantes numerique")
    # numeric_cols = df.select_dtypes(include=np.number).columns
    # for col in numeric_cols:
    #         plt.figure(figsize=(8, 3))
    #         sns.boxplot(x=df[col])
    #         plt.title(f"Boxplot - {col}")
    #         plt.show()

    # # Statistiques Variables numériques
    # print(df[numeric_cols].describe())
    
    # # Statistiques Variables catégorielles
    # print(df[categorial].describe())

    # print("\n")
    # # Fréquence de chaque classe
    # print(df["Mental_Health_Risk"].value_counts())
    # print("\n")
    # # Pourcentage de chaque classe
    # print(df["Mental_Health_Risk"].value_counts(normalize=True) * 100)


    # print("\n")

    # target_col="Mental_Health_Risk"

    # print("Distribution de la variable cible : \n")
    # plt.figure(figsize=(8, 5))
    # sns.countplot(x=df[target_col])
    # plt.title("Distribution de la variable cible")
    # plt.xlabel("Niveau de risque")
    # plt.ylabel("Nombre d'observations")
    # plt.xticks(rotation=45)
    # plt.show()


    # # the target with the numeric columns
    # print("\nthe target with the numeric columns")
    # for col in numeric_cols:
    #     if col != target_col:
    #         plt.figure(figsize=(8, 4))

    #         sns.boxplot(data=df, x=target_col, y=col)

    #         plt.title(f"{col} selon le niveau de risque")
    #         plt.xticks(rotation=45)
    #         plt.show()

    # # the target with the categorial columns
    # print("\nthe target with the categorial columns")
    # for col in categorial:
    #     if col != target_col and col!="Student_ID":
    #         plt.figure(figsize=(8, 4))

    #         sns.boxplot(data=df, x=target_col, y=col)

    #         plt.title(f"{col} selon le niveau de risque")
    #         plt.xticks(rotation=45)
    #         plt.show()


    # # Sélection des variables numériques
    # numeric_df = df.select_dtypes(include=np.number)

    # # Calcul des corrélations
    # corr_matrix = numeric_df.corr()

    # # Visualisation
    # plt.figure(figsize=(12, 8))

    # sns.heatmap(
    #     corr_matrix,
    #     annot=True,
    #     cmap="coolwarm",
    #     fmt=".2f",
    #     linewidths=0.5
    # )

    # plt.title("Matrice de corrélation")
    # plt.show()

      # 3. Calculate ANOVA scores
        # f_scores, p_values = f_classif(X, y)
    
        # # 4. Create a results table
        # results = pd.DataFrame({
        #     "Feature": X.columns,
        #     "F_Score": f_scores,
        #     "P_Value": p_values
        # })
    
        # # 5. Sort by F_Score
        # results = results.sort_values(by="F_Score", ascending=False)
    
        # print(results)