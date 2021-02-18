# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:15:37 2021

@author: Pascal
"""
import pandas as pd

class Standardization:
    
    def __init__(self, df_total):
        self.df_total = df_total

    #Functions for getting mean and std of each feature
    def calculate_Standardization(self):
        def mean_EAR(respondent):
            return df_means.loc[respondent]["EAR"]

        def mean_MAR(respondent):
            return df_means.loc[respondent]["MAR"]

        def mean_Circularity(respondent):
            return df_means.loc[respondent]["Circularity"]

        def mean_MOE(respondent):
            return df_means.loc[respondent]["MOE"]

        def std_EAR(respondent):
            return df_std.loc[respondent]["EAR"]

        def std_MAR(respondent):
            return df_std.loc[respondent]["MAR"]

        def std_Circularity(respondent):
            return df_std.loc[respondent]["Circularity"]

        def std_MOE(respondent):
            return df_std.loc[respondent]["MOE"]
        
        #Separating the rows which are "Alert" only
        df_alert = self.df_total[self.df_total["Y"] == 0] 
        
        #Creating separate dataframes for each participants's first, second and third "Alert" frame
        df_alert_1 = df_alert.iloc[0::240, :]
        df_alert_2 = df_alert.iloc[1::240, :]
        df_alert_3 = df_alert.iloc[2::240, :]
        
        #Merging them into one dataframe
        alert_first3 = [df_alert_1,df_alert_2,df_alert_3]
        df_alert_first3 = pd.concat(alert_first3)
        df_alert_first3 = df_alert_first3.sort_index()
        
        #Based on the first 3 "Alert" frames, calculating per participant the mean and std for each feature
        pd.options.mode.chained_assignment = None
        df_means = df_alert_first3.groupby("Participant")[["EAR", "MAR", "Circularity", "MOE"]].mean()
        df_std = df_alert_first3.groupby("Participant")[["EAR", "MAR", "Circularity", "MOE"]].std()
        
        #Adding respondent-wise mean and std for each feature to each row in the original dataframe
        self.df_total["EAR_mean"] = self.df_total["Participant"].apply(mean_EAR)
        self.df_total["MAR_mean"] = self.df_total["Participant"].apply(mean_MAR)
        self.df_total["Circularity_mean"] = self.df_total["Participant"].apply(mean_Circularity)
        self.df_total["MOE_mean"] = self.df_total["Participant"].apply(mean_MOE)

        self.df_total["EAR_std"] = self.df_total["Participant"].apply(std_EAR)
        self.df_total["MAR_std"] = self.df_total["Participant"].apply(std_MAR)
        self.df_total["Circularity_std"] = self.df_total["Participant"].apply(std_Circularity)
        self.df_total["MOE_std"] = self.df_total["Participant"].apply(std_MOE)
        self.df_total.head()
        print(self.df_total.shape)
        
        #Calculating now normalized features for each row in the original dataframe
        self.df_total["EAR_N"] = (self.df_total["EAR"] - self.df_total["EAR_mean"]) / self.df_total["EAR_std"]
        self.df_total["MAR_N"] = (self.df_total["MAR"] - self.df_total["MAR_mean"]) / self.df_total["MAR_std"]
        self.df_total["Circularity_N"] = (self.df_total["Circularity"] - self.df_total["Circularity_mean"]) / self.df_total["Circularity_std"]
        self.df_total["MOE_N"] = (self.df_total["MOE"] - self.df_total["MOE_mean"]) / self.df_total["MOE_std"]
        return self.df_total
