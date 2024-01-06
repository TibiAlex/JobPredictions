import numpy as np
import pandas as pd
import tensorflow as tf
import os
import math
from sklearn.model_selection import train_test_split
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import Levenshtein
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import confusion_matrix


class JobMatching:
    def __init__(self, k):
        # job_postings_path = './data/job_postings.csv'
        # companies_path = './data/company_details/companies.csv'
        # employee_path = './data/company_details/employee_counts.csv'
        # job_industries_path = './data/job_details/job_industries.csv'
        # industries_path = './data/maps/industries.csv'

        job_postings_path = './data/dataset.csv'
        companies_path = './data/dataset.csv'
        employee_path = './data/dataset.csv'
        job_industries_path = './data/dataset.csv'
        industries_path = './data/dataset.csv'

        self.job_postings_df = pd.read_csv(job_postings_path)
        self.companies_df = pd.read_csv(companies_path)
        self.employee_df = pd.read_csv(employee_path)
        self.job_industries_df = pd.read_csv(job_industries_path)
        self.industries_df = pd.read_csv(industries_path)
        
        def levenshtein_distance(s1, s2):
            return Levenshtein.distance(s1, s2)
        
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=levenshtein_distance, weights='distance', algorithm='ball_tree')
        self.nca = NeighborhoodComponentsAnalysis(random_state=42)
        self.minvalsal = self.job_postings_df['min_salary'].min()
        self.maxvalsal = self.job_postings_df['min_salary'].max()
        self.minvalemp = 0
        self.maxvalemp = 0
        self.reversed_dict = None
        self.copy_job = copy.deepcopy(self.job_postings_df)
        self.label_encoder = LabelEncoder()
        self.imputer = KNNImputer(n_neighbors=k, metric='nan_euclidean', weights='distance')

    def create_dataset(self):
        companies_values = []
        companies_dict = {row['company_id']: row['name'] for idx, row in self.companies_df.iterrows()}
        for index, row in self.job_postings_df.iterrows():
            if math.isnan(row['company_id']):
                companies_values.append(np.nan)
            else:
                if row['company_id'] not in companies_dict.keys():
                    companies_values.append(np.nan)
                else:
                    companies_values.append(companies_dict[row['company_id']])
        self.job_postings_df.insert(loc=1, column='company', value=companies_values)

        companies_employee_value = []
        companies_employee_dict = {row['company_id']: row['employee_count'] for idx, row in self.employee_df.iterrows()}
        for index, row in self.job_postings_df.iterrows():
            if math.isnan(row['company_id']):
                companies_employee_value.append(np.nan)
            else:
                if row['company_id'] not in companies_employee_dict.keys():
                    companies_employee_value.append(np.nan)
                else:
                    companies_employee_value.append(companies_employee_dict[row['company_id']])
        self.job_postings_df.insert(loc=1, column='employee_count', value=companies_employee_value)

        industrie_value = []
        job_industries_dict = {row['job_id']: row['industry_id'] for idx, row in self.job_industries_df.iterrows()}
        industries_dict = {row['industry_id']: row['industry_name'] for idx, row in self.industries_df.iterrows()}
        for index, row in self.job_postings_df.iterrows():
            if math.isnan(row['job_id']):
                industrie_value.append(np.nan)
            else:
                if row['job_id'] not in job_industries_dict.keys():
                    industrie_value.append(np.nan)
                else:
                    if job_industries_dict[row['job_id']] not in industries_dict:
                        industrie_value.append(np.nan)
                    else:
                        industrie_value.append(industries_dict[job_industries_dict[row['job_id']]])
        self.job_postings_df.insert(loc=1, column='industries', value=industrie_value)
        self.job_postings_df = self.job_postings_df[['min_salary', 'remote_allowed', 'formatted_experience_level', 'location', 'employee_count', 'industries', 'company']]
        self.job_postings_df.to_csv('./data/dataset.csv', index=False)
        return self.job_postings_df

    def preprocessing_data(self, tp, show_uniques, jb_df):
        if (show_uniques == True):
            values = jb_df.nunique().tolist()
            column_names = jb_df.columns.tolist()

            plt.figure(figsize=(16, 6))
            ax = sns.barplot(x=column_names, y=values, palette="viridis")

            for i, value in enumerate(values):
                ax.text(i, value + 0.5, str(value), ha='center', va='bottom')

            plt.xlabel("Category")
            plt.ylabel("Values")
            plt.title("Barplot for 7 Values")

            plt.show()

        companies_idx_dict = {title: idx for idx, title in enumerate(jb_df['company'].unique())}
        self.companies_reversed_dict = {value: key for key, value in companies_idx_dict.items()}
        jb_df['company'].replace(companies_idx_dict, inplace=True)
        def replaceCompanyValWithNan(val):
            if val == companies_idx_dict[np.nan]:
                return np.nan
            else:
                return val
        jb_df['company'] = jb_df['company'].apply(replaceCompanyValWithNan, 1)
#         jb_df['company'] = self.label_encoder.fit_transform(jb_df['company'])

        ep_dict = {ep: idx for idx, ep in enumerate(jb_df['formatted_experience_level'].unique())}
        self.ep_reversed_dict = {value: key for key, value in ep_dict.items()}
        jb_df['formatted_experience_level'].replace(ep_dict, inplace=True)
        def replaceValWithNan(val):
            if val == ep_dict[np.nan]:
                return np.nan
            else:
                return val
        jb_df['formatted_experience_level'] = jb_df['formatted_experience_level'].apply(replaceValWithNan, 1)
#         jb_df['formatted_experience_level'] = self.label_encoder.fit_transform(jb_df['formatted_experience_level'])

        if (tp == 'init_dataset'):
            industry_dict = {ind: idx for idx, ind in enumerate(jb_df['industries'].unique())}
            self.industry_reversed_dict = {value: key for key, value in industry_dict.items()}
            jb_df['industries'].replace(industry_dict, inplace=True)
            def replaceIndValWithNan(val):
                if val == industry_dict[np.nan]:
                    return np.nan
                else:
                    return val
            jb_df['industries'] = jb_df['industries'].apply(replaceIndValWithNan, 1)

        jb_df['location'] = self.label_encoder.fit_transform(jb_df['location'])

        def replaceNaNRemote(val):
            if math.isnan(val):
                return 0
            else:
                return 1
        jb_df['remote_allowed'] = jb_df['remote_allowed'].apply(replaceNaNRemote, 1)

        jb_df = pd.DataFrame(self.imputer.fit_transform(jb_df), columns=jb_df.columns)

#         def minmaxvalue(val):
#             if not math.isnan(val):
#                 return (val - self.minvalsal) / (self.maxvalsal - self.minvalsal)
#         jb_df['min_salary'] = jb_df['min_salary'].apply(minmaxvalue, 1)

#         self.minvalemp = jb_df['employee_count'].min()
#         self.maxvalemp = jb_df['employee_count'].max()
#         def minmaxvalueen(val):
#             if not math.isnan(val):
#                 return (val - self.minvalemp) / (self.maxvalemp - self.minvalemp)
#         jb_df['employee_count'] = jb_df['employee_count'].apply(minmaxvalueen, 1)

        jb_df['remote_allowed'] = jb_df['remote_allowed'].astype(int)
        jb_df['formatted_experience_level'] = jb_df['formatted_experience_level'].astype(int)
        jb_df['location'] = jb_df['location'].astype(int)
        jb_df['industries'] = jb_df['industries'].astype(int)
        jb_df['company'] = jb_df['company'].astype(int)

        jb_df.to_csv('./data/dataset_preprocessed.csv', index=False)

        scaler = StandardScaler()


        if (tp == 'init_dataset'):
            X = scaler.fit_transform(jb_df[['min_salary', 'remote_allowed', 'formatted_experience_level', 'location', 'employee_count', 'company']])
            y = jb_df['industries']
            return X, y
        else:
            X = scaler.fit_transform(jb_df[['min_salary', 'remote_allowed', 'formatted_experience_level', 'location', 'employee_count', 'company']])
            return X

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def trainKNN(self, X_train, y_train):
        self.knn_classifier.fit(X_train, y_train)
    
    def predictWithKNN(self, X_test):
        y_pred = self.knn_classifier.predict(X_test)
        return y_pred
    
    def predictForOnlyOneInput(self, input_dataframe, new_queue_X, new_queue_y):
        scaled_df_X = self.preprocessing_data('raw', False, input_dataframe)
        pred = self.knn_classifier.predict(scaled_df_X)

        new_queue_X.append(scaled_df_X)
        new_queue_y.append(pred)

        return pred

    def getScore(self):
        X, y = self.preprocessing_data("raw", False)
        print(X.shape)
        print(y.shape)
        return cross_val_score(self.knn_classifier, X, y, cv=5, n_jobs=-1)
    
    def getRMSQ(self):
        X, y = self.preprocessing_data("raw", False)
        return cross_val_score(self.knn_classifier, X, y, cv=5, scoring='root_mean_squared_error', n_jobs=-1)
    
#     def preprocessing_pp_data(self):
#         pp_path = '/kaggle/input/df-tibi-v2/dataframe_formatted.csv'
#         self.pp_df = pd.read_csv(pp_path)
    
#         self.pp_df['title'] = self.label_encoder.fit_transform(self.pp_df['title'])
#         self.pp_df['formatted_work_type'] = self.label_encoder.fit_transform(self.pp_df['formatted_work_type'])
#         self.pp_df['location'] = self.label_encoder.fit_transform(self.pp_df['location'])
#         self.pp_df['application_type'] = self.label_encoder.fit_transform(self.pp_df['application_type'])
#         self.pp_df['speciality'] = self.label_encoder.fit_transform(self.pp_df['speciality'])
#         self.pp_df['formatted_experience_level'] = self.label_encoder.fit_transform(self.pp_df['formatted_experience_level'])
#         self.pp_df['employee_count'] = self.label_encoder.fit_transform(self.pp_df['employee_count'])
#         self.pp_df['job_industries'] = self.label_encoder.fit_transform(self.pp_df['job_industries'])
#         self.pp_df['remote_allowed'] = self.pp_df['remote_allowed'].astype(int)
        
#         print(self.pp_df)
        
#         X_train, X_test, y_train, y_test = train_test_split(self.pp_df[['employee_count', 'remote_allowed', 'formatted_work_type', 'location', 'application_type', 'speciality', 'formatted_experience_level']], self.pp_df['job_industries'], test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#         return X_train_scaled, X_test_scaled, y_train, y_test
#         return X_train, X_test, y_train, y_test

#     def calculateAccuracy(self, y_test, y_pred, X_train, y_train):
# #         self.nn.fit(self.job_postings_df)
#         valid = 0
#         for idx, x in enumerate(y_test):
# #             indexes = self.job_postings_df[self.job_postings_df['title'] == y_pred[idx]].index
# #             rows = self.job_postings_df[self.job_postings_df['title'] == y_pred[idx]]
# #             indexes = y_train[y_train['title'] == y_pred[idx]].index
#             indexes = y_train.index[y_train == y_pred[idx]].tolist()
#             print(indexes)
#             print(X_train.loc[indexes[0]])
#             neighbours = self.knn_classifier.kneighbors([X_train.loc[indexes[0]]], n_neighbors=10, return_distance=False)
#             neighbours = (neighbours.tolist())[0]
#             print(neighbours)

#             resp = list(map(lambda x: self.job_postings_df['title'].loc[x], neighbours))
# #             print(resp)
#             if list(y_test)[idx] in resp:
#                 valid = valid + 1
#             break
#         return valid * 1.0 / len(list(y_test))
