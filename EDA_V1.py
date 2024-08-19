from scipy.stats import skew, yeojohnson, shapiro
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats
from pyod.models.mad import MAD
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from gower import gower_matrix
from sklearn.manifold import TSNE
# import hdbscan



class Initial_Analysis:
    def __init__(self, data, project_name = None, target = None):
        self.data = data.copy()
        self.plot_null = None
        self.outliers = None
        self.outliers_dict = None
        self.outliers_df = None
        self.project_name = project_name
        self.skew_threshold = None
        self.original_data = data.copy()
        
        self.skew_before = {}
        self.skew_after = {}
        self.highly_skewed = []
       
        
        self.encoded_features = {}
        self.ohe_original = {}
        self.le = LabelEncoder()
        self.oe = OrdinalEncoder()
        self.ohe = OneHotEncoder()
        
        self.outliers_removed = None
        self.mad_outliers_removed = None
        
        self.home = Path.home()
        self.target = target
        
        
        self.dbscan_params = None
        self.silhouette_threshold = None
        self.stop_cluster = False
        
    
    def initial_analysis(self, poly_feat_data = None):
        if poly_feat_data is not None:
            self.data = poly_feat_data
            print('\033[1m' +'\nBeginning Analysis of Polynomial Relationships\n' + '\033[0m')
        else:
            print('\033[1m' + 'Beginning Initial Analysis\n' + '\033[0m')
        
        
        print('\033[1m' +'Summary Stats:\n'+ '\033[0m')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(round(self.data.describe().T, 2))
        print('\nIf there is significant disparity between mean, median and mode it is an indicator that there may be a class imbalance\n')
        modes = {}
        for feature in self.data.columns:
            mode_values = self.data[feature].mode()
            mode_count = self.data[feature].eq(mode_values[0]).sum()
            total = self.data[feature].count()
            modes[feature] = mode_values
            mode_percent = round((mode_count/total) * 100, 2)
            
            if len(mode_values.values) == self.data.shape[1]:
                print(f'{feature} has no modes - all values are unique\n')
            else:
                print(f'{feature} has mode(s) of:')
                print(f'{mode_values.values}, which makes up {mode_percent}% of non-NaN values\n')
                  
                

        print('\033[1m' +'\nNumber of Unique Values:\n'+ '\033[0m')
        for feature in self.data.columns:
            num_unique = len(self.data[feature].value_counts())
            total = self.data.shape[0]
            print(f'{feature} has {num_unique} unique values out of {total}')

        
        print('\033[1m' +'\nNumber of Null Values:\n'+ '\033[0m')
        for feature in self.data.columns:
            num_null = self.data[feature].isnull().sum()
            total = self.data.shape[0]
            percentage = ((num_null/total)*100).round(1)
            print(f'{feature} has {num_null} null values out of {total}: {percentage}%')
            if percentage >= 50:
                remove = input(f'\n{feature} has significant number of missing values. Remove feature? Y/N ')
                while True:
                    if remove.lower() == 'y':
                        print(f'Removing {feature}\n')
                        self.data = self.data.drop(feature, axis = 1)
                        break
                    if remove.lower() == 'n':
                        print('Continuing\n')
                        break
                    else: 
                        remove = input('\nInvalid input - Y/N are only accepted inputs ')
                        
        print('\033[1m' +'\nFeature Data Types:\n'+ '\033[0m')
        for feature in self.data.columns:
            dtype = self.data[feature].dtypes
            print(f'{feature} is {dtype} data')
            numerical_types = ('int', 'float', 'complex')
            categorical_options = ['1) Drop', '2) Label Encoder', '3) Ordinal Encoder', '4) One Hot Encoder']
            
            ohe_column_names = []
            
            if dtype not in numerical_types:
                unique_values = self.data[feature].unique()
                while True:
                    print(f'{feature} has unique values of {unique_values}')
                
                    encode = input(f'{feature} appears categorical. Please select from'
                                   f' {categorical_options} \n')
                    if '1' in encode:
                        self.data = self.data.drop(feature, axis = 1)
                        break
                    elif '2' in encode:
                        self.encoded_features = {feature: {}}
                        encoded_data = self.le.fit_transform(self.data[feature])
                        self.data[feature] = encoded_data
                        self.encoded_features[feature]['encoder'] = self.le
                        break
                    elif '3' in encode:
                        self.encoded_features = {feature: {}}
                        data_to_encode = self.data[feature].values.reshape(-1, 1)
                        self.data[feature] = self.oe.fit_transform(data_to_encode)   
                        self.encoded_features[feature]['encoder'] = self.oe
                        break
                    elif '4' in encode:
                        data_to_encode = self.data[feature].values.reshape(-1, 1)
                        encoded_data = self.ohe.fit_transform(data_to_encode)
                        encoded_array = encoded_data.toarray()
                        encoded_columns = self.ohe.get_feature_names_out([feature])
                        self.ohe_original = {feature: []}
                        for column in encoded_columns:
                            remove = f'{feature}_'
                            column = column.replace(remove, '')
                            self.encoded_features = {column: {}}
                            ohe_column_names.append(column)
                            self.encoded_features[column]['encoder'] = self.ohe
                            self.ohe_original[feature].append(column)
                            
                        encoded_df = pd.DataFrame(encoded_array, columns=ohe_column_names)
                       # print(ohe_column_names)
                        
                        # Concatenate the encoded DataFrame with the original DataFrame
                        self.data.drop(columns=[feature], inplace=True)
                        self.data = pd.concat([self.data, encoded_df], axis=1)
                        
                        break
                    elif encode == '':
                        return 'Exiting'
        
        print('\033[1m' +'\nFinding Outliers Based on Z-Score and Median Absolute Deviation (MAD):\n'+ '\033[0m')
        outlier_dicts = []
        outliers = {}
        outlier_indices = []
        
        mad_outlier_dicts = []
        mad_outliers = {}
        mad_outlier_list = []
       
        warnings.resetwarnings()
        
        ohe_features = []
        for original_feature, encoded_features in self.ohe_original.items():
            for encoded_feature in encoded_features:
                ohe_features.append(encoded_feature)
        
        print(ohe_features)
        
        for feature in self.data.columns:
            if feature not in self.encoded_features and feature not in ohe_features:
                dtype = self.data[feature].dtypes
            
            #checks data type
                if dtype != 'object':
                    threshold = 3
                    
                    z_score_data = self.data[feature].dropna()
                    z_score = stats.zscore(z_score_data)
                    
                    
                    outliers[feature] = {'Positive Outliers': [], 
                                         'Negative Outliers': []}
                    
                    detector = MAD()
                    mad_data = self.data[feature].values.reshape(-1, 1)
                    detector.fit(mad_data)
                    outlier_scores = detector.decision_scores_
                    mad_outlier_indices = np.where(outlier_scores > threshold)[0]
                    
                    mad_outlier_list.append(np.array(mad_outlier_indices))
                    outliers[feature]['MAD'] = len(mad_outlier_indices)

                    for index, value in enumerate(z_score):
                        if abs(value) >= threshold:
                            outlier_indices.append(z_score_data.index[index])

                            if value > 0:
                                outliers[feature]['Positive Outliers'].append(round(value, 2))
                            else:
                                outliers[feature]['Negative Outliers'].append(round(value, 2))
                    
                #removes any features with no outliers
                    if len(outliers[feature]['Positive Outliers']) == 0 and len(outliers[feature]['Negative Outliers']) == 0:
                        del outliers[feature]
            
        #filter non-zero arrays to be concatenated  
        filtered_outlier_list = [arr for arr in mad_outlier_list if arr.size > 0]

        if filtered_outlier_list:
            concatenated_outliers = np.concatenate(filtered_outlier_list)
            unique_outlier_indices = np.unique(concatenated_outliers)
            self.mad_outliers_removed = self.data.drop(unique_outlier_indices)
        else:
            print('No non-empty arrays found for concatenation.')

        
        if outliers:
            for feature, outlier_counts in outliers.items():
                outlier_dicts.append({
                    'Feature': feature,
                    'Positive Outliers': len(outlier_counts['Positive Outliers']),
                    'Negative Outliers': len(outlier_counts['Negative Outliers']),
                    'MAD': outliers[feature]['MAD']
                    })
                self.outliers = outliers
                
                #new stuff here
                self.outliers_removed = self.data.drop(outlier_indices)
                    
                total_mad_outliers = outliers[feature]['MAD']
                total_outliers = len(outlier_counts['Positive Outliers']) + len(outlier_counts['Negative Outliers'])
                print(f'{feature} has {total_outliers} total outliers based on Z-Score')
                print(f'and {total_mad_outliers} based on MAD')
                outlier_df = pd.DataFrame(outlier_dicts)
                
        if self.mad_outliers_removed.shape[0] == self.data.shape[0]:
            return 'Failed to remove outliers'
        else:
            pass
            #print('\nMAD outliers removed successfully')
                
        if self.outliers_removed.shape[0] == self.data.shape[0]:
            return 'Failed to remove Z-Score outliers'
        else:
            pass
            #print('Z-Score outliers removed successfully')
        
        def plot_outliers(outlier_df, mad = False):
            print('\033[1m' +'\nPlotting Number of Outliers:\n' + '\033[0m')
            
            home = Path.home()
            if not mad:
                non_zero_outliers = outlier_df[(outlier_df['Positive Outliers'] != 0) | (outlier_df['Negative Outliers'] != 0)]
            else:
                non_zero_outliers = outlier_df[outlier_df['MAD'] != 0]
                
            if not non_zero_outliers.empty:
                if not self.project_name:
                    project_name = input("What is this project called?")
                    self.project_name = project_name
                else:
                    project_name = self.project_name  
                    outlier_df = non_zero_outliers
                    plt.figure(figsize=(10, 6))
                    
                name = input('What are you plotting the number of? ')
                file_name = name.replace(' ', '_')
                if not mad:
                    plt.barh(outlier_df['Feature'], outlier_df['Positive Outliers'], label='Positive Outliers')
                    plt.barh(outlier_df['Feature'], outlier_df['Negative Outliers'], label='Negative Outliers', left=outlier_df['Positive Outliers'])
                    title = 'Positive and Negative '
                    file_path = f'{self.home}/Desktop/Coding/projects/{project_name}/figures/outliers/z_score_outliers_{file_name}'
                        
                else:
                    plt.barh(outlier_df['Feature'], outlier_df['MAD'], label='Outliers')
                    title = 'MAD '
                    file_path = f'{self.home}/Desktop/Coding/projects/{project_name}/figures/outliers/mad_outliers_{file_name}'
                    
                plt.title(f"{title} Outliers per Feature for {name}")     
                plt.ylabel("Features")
                plt.xlabel("Number of Outliers")

                plt.legend()
                fig = plt.gcf()
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)
                plt.savefig(file_path, bbox_inches='tight') 
                plt.show()
            else:
                print('No outliers')
        
        if outliers:
            plot_outliers(outlier_df)
            plot_outliers(outlier_df, mad = True)
            #remove_outliers = input('Would you like to remove outliers? Y/N ')
            while True:
                remove_outliers = input('Would you like to remove outliers? Y/N ')
                if 'y' in remove_outliers.lower():
                    self.outlier_removal()
                    self.outlier_removal(mad = True)
                    break
                elif 'n' in remove_outliers.lower():
                    break
                else:
                     print('Invalid Input')
        
        #return self.outliers_removed, self.mad_outliers_removed
    
        print('\033[1m' +'\nCorrelation Heatmap:\n'+ '\033[0m')
        
        num_features = self.data.shape[1]
        
        figsize = (min(10 + num_features * 0.2, 20), min(8 + num_features * 0.15, 15))
        corr_matrix = self.data.corr()
        plt.figure(figsize = figsize)
        fontsize = min(12, max(8, 120 // num_features))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": fontsize})
        plt.title(f'Correlation of features for {self.project_name}')
        file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/correlation_matrix/linear_relationships'
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight') 
        plt.show()
        
        print('\033[1m' +'Checking for Linear Multicollinearity\n'+ '\033[0m')
        
        vif_features = []
        for feature in self.data.columns:
            if self.data[feature].dtypes != 'object':
                vif_features.append(feature)
                
        vif_data = self.data[vif_features].dropna()
        vif = pd.DataFrame()
        vif["Features"] = vif_data.columns
        vif["VIF Factor"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
        
        low_threshold = 2
        moderate_threshold = 5
        high_threshold = 10
        
        def categorise_collinearity(vif_value):
            if vif_value <= low_threshold:
                return 'Low'
            elif low_threshold < vif_value <= moderate_threshold:
                return 'Moderate'
            elif moderate_threshold < vif_value <= high_threshold:
                return 'High'
            else:
                return 'Very High'

            
        def calculate_correlation(feature_name):
            other_features = [feature for feature in self.data.columns if feature != feature_name]
            vif_data_without_target = self.data[other_features]
            vif_without_target = pd.DataFrame()
            vif_without_target["Features"] = vif_data_without_target.columns
            vif_without_target["VIF Factor"] = [variance_inflation_factor(vif_data_without_target.values, i) for i in range(vif_data_without_target.shape[1])]
            correlation_with_target = self.data[other_features].corrwith(self.data[target_feature])

            # Analyse correlations or VIF without target to understand contributing factors
            print(f"Correlation with '{target_feature}':")
            print(correlation_with_target)
            print(f"\nVIF without '{target_feature}':")
            print(vif_without_target)
            correlation_results = self.data.corr()[feature_name]
            return correlation_results
        
        def calculate_correlation(threshold, vif, data):
            target_features = []
            for index, row in vif.iterrows():
                if row['VIF Factor'] > threshold:
                    target_features.append(row['Features'])
            print(f'\nAnalysing Collinearity of {target_features}')  
            
            #selecting features and data
            for target_feature in target_features:
                data = self.data.dropna()
                features = [feature for feature in data.columns if data[feature].dtypes != 'object']
                other_features = [feature for feature in features if feature != target_feature]
                
                #calculating vif with target 
                vif_data_with_target = data[features]
                vif_with_target = pd.DataFrame()
                vif_with_target["Features"] = vif_data_with_target.columns
                vif_with_target[f"VIF Factor With {target_feature}"] = [
                    variance_inflation_factor(vif_data_with_target.values, i) for i in range(vif_data_with_target.shape[1])
                ]
                
                #calculating vif without target
                vif_data_without_target = data[other_features]
                vif_without_target = pd.DataFrame()
                vif_without_target["Features"] = vif_data_without_target.columns
                vif_without_target[f"VIF Factor Without {target_feature}"] = [variance_inflation_factor(vif_data_without_target.values, i) for i in range(vif_data_without_target.shape[1])]
                
                #correlatiion with target
                correlation_with_target = data[other_features].corrwith(data[target_feature])
                
                #combining results
                combined_df = pd.concat([correlation_with_target, 
                                         vif_with_target.set_index('Features'),
                                        vif_without_target.set_index('Features')], axis=1)
                
                combined_df = combined_df.drop(target_feature)
                
                combined_df.columns = [f'Correlation with {target_feature}', f'VIF Factor With',
                                       f'VIF Factor Without']
                print(f'\nVIF Analysis Table for {target_feature}:')
                print(combined_df)
                
                
                log_df = combined_df.copy()
                log_df = log_df.drop(f'Correlation with {target_feature}', axis =1)
                
                vif_columns = ['VIF Factor With', 'VIF Factor Without']
                log_df[vif_columns] = log_df[vif_columns].apply(np.log)
                
                num_features = log_df.shape[1]
                figsize = (min(10 + num_features * 0.2, 20), 
                           min(8 + num_features * 0.15, 15))
                
            
                plt.figure(figsize = figsize)
                
                plt.bar(log_df.index, log_df['VIF Factor With'], label = f'VIF factors with {target_feature}')
                plt.bar(log_df.index, log_df['VIF Factor Without'], label = f'VIF factors without {target_feature}')
                
                plt.xlabel('Features')
                plt.ylabel('Logarithmic VIF')
                plt.title(f'Comparison of Logarithmic VIF with and without {target_feature}')
                plt.xticks(rotation=90)
                plt.legend()
                plt.tight_layout()
                file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/linear_vif_analysis/{target_feature}'
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)
                plt.savefig(file_path, bbox_inches='tight') 

                plt.show()
            
            # Analyse correlations or VIF without target to understand contributing factors
            correlation_results = self.data.corr()[target_feature]
            return correlation_results
                

        # Add a new column 'CollinearityDegree' based on VIF values
        vif['Collinearity Degree'] = vif['VIF Factor'].apply(categorise_collinearity)
        
        
        print(vif)
        print('\nHigh to Very High values indicate that further processing is required as it may impact the model\n')
        
        
        vif_analysis = input('Would you like to analyse the VIF results? Y/N ')
        
        while True:
            if vif_analysis.lower() == 'y':
                threshold = input('Above what VIF threshold would you like perform analysis? ')
                try:
                    threshold = float(threshold)
                    calculate_correlation(threshold, vif, self.data)
                    break
                except ValueError:
                    print(f'{threshold} could not be converted - numeric inputs only')
                    
            elif vif_analysis.lower() == 'n':
                break
            else:
                vif_analysis = input(f'{vif_analysis} is an invalid input please enter Y/N ')
    
    
        #analysis of the class distributributions
        print('\033[1m' +'\nCalculating Class Balance Metrics and Feature Skewness:\n'+ '\033[0m')
        
        #definitions and formatting
        binary = '\033[1m' +'binary'+ '\033[0m'
        multi = '\033[1m' +'multiple'+ '\033[0m'
        Gini = '\033[1m' +'Gini Coefficient'+ '\033[0m'
        Shannon = '\033[1m' +'Shannon Entropy'+ '\033[0m'
        Ratio = '\033[1m' +'Class Imbalance Ratios'+ '\033[0m'
        Skew = '\033[1m' +'Skewness'+ '\033[0m'
        Linear = '\033[1m' +'Linear Regression'+ '\033[0m'
        KNN ='\033[1m' +'k-Nearest Neighbors'+ '\033[0m' 
        Neural = '\033[1m' +'Neural Networks'+ '\033[0m' 
        
        #definitions and information about metrics used for eda
        print(f'{Skew}: Provides information about the asymmetry in the distribution '
              f'of values within a class or feature. Models like {Linear}, {KNN}, '
              f'and {Neural} might struggle with skewed data, as they assume a more '
              f'symmetrical distribution for accurate predictions. When a feature\'s |{Skew}| > 0.7 ' 
              'you may need to consider transforming those features\n')
        
        print(f'{Gini}: Useful when assessing class imbalance for features with {binary} classes, ' 
              f'but gets more complex and difficult for {multi} class features. '
              'Measures inequality among values in a frequency distribution.\n')
        
        print(f'{Shannon}: Effective when seeking a measure of uncertainty or disorder in the class labels, '
              'especially in scenarios where understanding the information gain (entropy) is essential. '
              f'Suitable at assessing imbalance for both {binary} and {multi} class features. '
              'Measures uncertainty or disorder in a set of class labels.\n')
        
        print(f'{Ratio}: Helpful for a quick assessment of imbalance ' 
              f'in {binary} classification problems when comparing the counts of ' 
              'the majority and minority classes, providing a simple and intuitive ' 
              f'understanding of the imbalance a value of 1 indicates that a {binary} class feature is balanced. '
              f'However, struggles when {multi} classes present in feature\n')
        
        print(f'Note: {Gini} and {Shannon} do not directly measure class imbalance, '
             f'however, a {Gini} closer to 1 suggests greater seperation of classes, and this indirectly may indicate balance; '
             f'whereas a lower {Shannon} indicates more disorder and therefore could indicate that the classes are more balanced.\n')
        
        #variables to store results
        balance_metrics = {}
        skewed= {}
        highly_skewed = {}
        
        qq_response = ['all', 'selected', 'none']
        
        
        print('QQ Plots compare a dataset against a normal distribution. If normally' 
              ' distributed the blue dots will lay along the reference line (or nearby)' 
              ' curvature or deviations away from the red line indicate that the distribution' 
              ' is not normal')
        
        while True:
            visualise_all_qq = input(f'Would you like to visualise the QQ plots of' 
                                     f' all features? Choose from {qq_response} ')
            if 'a' in visualise_all_qq.lower():
                print('Ready to Plot QQ')
                visualise_all_qq = 'a'
                break
            elif 's' in visualise_all_qq.lower():
                print('Plotting Class Distributions')
                visualise_all_qq = 's'
                break
            elif 'n' in visualise_all_qq.lower():
                visualise_all_qq = 'n'
                break
            else:
                print(f'{visualise_all_qq} is an invalid input. Please enter Y/N')
        
        if self.ohe_original:
            decoded_data = []
            for original_feature, ohe_feature in self.ohe_original.items():
                ohe_feats = ohe_feature
                encoded_data = self.data[ohe_feature]
                for _, encoded_rows in encoded_data.iterrows():
                    index = np.where(encoded_rows == 1)[0][0]  # Get the index of '1'
                    decoded_data.append(ohe_feature[index])
                    original_values = pd.Series(decoded_data).value_counts()

                plt.bar(original_values.index, original_values)
                plt.xlabel(original_feature)
                plt.ylabel('Count')
                plt.title(f'Distribution of {original_feature}')
                file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/class_dist/histograms/{original_feature}'
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)
                plt.savefig(file_path, bbox_inches='tight') 
            plt.show()
            
            qq_title = original_feature
            if visualise_all_qq == 'a':
                self.QQ_plot(original_values, original_feature)
            if visualise_all_qq == 's':   
                while True:
                    visualise_qq = input(f'Would you like to visualise the QQ for {original_feature}? Y/N ')
                    if visualise_qq.lower()== 'y':
                        self.QQ_plot(original_values, original_feature)
                        break
                    elif visualise_qq.lower() == 'n':
                        break
                    else:
                        print('Invalid Input. Please enter Y/N')

                
        else:
            ohe_feats = []
        
        for feature in self.data.columns:
            if feature not in ohe_feats:
                if feature in self.encoded_features:
                    encoder = self.encoded_features[feature]['encoder']
                #print(self.ohe_original.values())
                    if feature not in ohe_feats:
                        try:
                            decoded_data = encoder.inverse_transform(self.data[feature])
                            values = pd.Series(decoded_data).value_counts()
                        except ValueError:
                            data_to_decode = self.data[feature].values.reshape(-1, 1)
                            decoded_data = encoder.inverse_transform(data_to_decode)
                            decoded_data = decoded_data.flatten()
                            values = pd.Series(decoded_data).value_counts()
                        
                    if self.ohe_original:
                    #print(self.ohe_original)
                        decoded_data = []
                        for original_name, decoded_columns in self.ohe_original.items():
                            encoded_data = self.data[decoded_columns]
                            for _, encoded_rows in encoded_data.iterrows():
                                index = np.where(encoded_rows == 1)[0][0]  # Get the index of '1'
                                decoded_data.append(decoded_columns[index])
                        values = pd.Series(decoded_data).value_counts()    

                else:
                    values = self.data[feature].value_counts()
             
                                    
                plt.bar(values.index, values)
                plt.xlabel(feature)
                plt.ylabel('Count')
                plt.title(f'Distribution of {feature}')
                file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/class_dist/histograms/{feature}'
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)
                plt.savefig(file_path, bbox_inches='tight') 
                plt.show()
            

                qq_title = f'{feature}'
                if visualise_all_qq.lower() == 'a':
                    self.QQ_plot(self.data[feature], qq_title) 
                elif visualise_all_qq.lower() == 's':
                    while True:
                        visualise_qq = input(f'Would you like to visualise the QQ for {feature}? Y/N ')
                        if visualise_qq.lower()== 'y':
                            self.QQ_plot(self.data[feature], qq_title)
                            break
                        elif visualise_qq.lower() == 'n':
                            break
                        else:
                            print('Invalid Input. Please enter Y/N')

            #calculating metrics
            
                class_labels = self.data[feature]
                class_proportions = class_labels.value_counts(normalize=True)
            
                shannon_entropy = scipy.stats.entropy(class_labels.value_counts(normalize=True), base=2)     
                gini_coefficient = 1 - (class_proportions ** 2).sum()          
                imbalance_ratio = values.max() / values.min()            
                skew_score = self.data[feature].skew()

            #printing metrics under graph
                print('\033[1m'+ f'{feature}'+ '\033[0m')
                print(f"Gini Coefficient: {gini_coefficient:.4f}")
                print(f"Shannon Entropy: {shannon_entropy:.4f}")
                print(f"Imbalance Ratio: {imbalance_ratio:.4f}")
                print(f'Skewness: {skew_score:.4f}\n')
            
            
                balance_metrics[feature] = {'Gini Coef': gini_coefficient, 
                                           'Entropy': shannon_entropy,
                                           'Imbalance Ratio': imbalance_ratio}
            if skew_score != 0:
                skewed[feature] = skew_score
        
        
        if skewed:
            while True:
                threshold = input('\nWhat is your threshold over which you would like to transform the data? ')
                try:
                    self.skew_threshold = float(threshold)
                    if isinstance(self.skew_threshold, float):  # Check if the conversion to float was successful
                        break
                except ValueError:
                    print(f'{threshold} is an invalid input: Please enter a numeric value')
        
        
        for feature, skew_score in skewed.items():
            if feature not in self.encoded_features:
                if abs(skew_score) >= self.skew_threshold:
                    highly_skewed[feature] = skew_score
        
        
        if self.ohe_original:
            for feature_name, feature_values in self.ohe_original.items():
                features = tuple(feature_values)
                for feature in features:
                    if feature in highly_skewed:
                        del highly_skewed[feature]
        
        if highly_skewed:
            print(f'\nReady to transform: {highly_skewed}')
            self.transform_data(highly_skewed)

            
    def transform_data(self, highly_skewed):
        #ignoring warnings 
        warnings.filterwarnings("ignore")
        
        #information about transformation, the statistical tests, and hypothesis testing
        print('\033[1m'+ '\nPreparing to Transform Highly Skewed Features\n'+ '\033[0m')
        H0 = 'H0: The Transform was unsuccessful at making the data more normally distributed'
        H1 = 'H1: The Transform was successful at making the data more normally distributed'
        datasets = {}
        print(H0)
        print(H1 +'\n')
        
        print('The Shapiro-Wilk test is sensitive to small sample sizes but may ' 
              'lack accuracy with non-normal distributions; the Kolmogorov-Smirnov' 
              ' test is versatile but less powerful for detecting deviations from'
              ' normality; while the Anderson-Darling test is effective for various' 
              ' sample sizes but might be influenced by extreme values.\n')
        
        test = '\033[1m' +'Test Statistic'+ '\033[0m' 
        p_val = '\033[1m' +'P. Value'+ '\033[0m' 
        crit_val = '\033[1m' +'Critical Value'+ '\033[0m' 
        
        print(f'{test}: The value determined by the statistical test and is used to determine P Value\n')
        print(f'{p_val}: The probability of a more extreme case - for this programme if p < alpha then accept H0; ' 
              'however, p < alpha would usually result in the null being rejected. '
              'Due to the set up of this test it is reversed\n')
        print(f'{crit_val}: Critical values are specific values associated with a ' 
              'chosen significance level (alpha) for a given statistical test.\n')

        data = self.data

         #define significance level before the for loop 
        while True:
            alpha_list = [15,  10,   5,   2.5,  1 ]
            alpha = input(f'Please select a significance level out of the following {alpha_list} to use to '
                            'test for normality of the distribution? ')
            try:
                alpha = float(alpha)
                alpha = alpha/100
                if alpha >= 0 and alpha <= 1:
                    break
                else:
                    print('Invalid input: Alpha should be a number between 1 and 15')
            except ValueError:
                print('Invalid input: Please enter a valid numeric value for alpha.')
        
        
        visualise_dist = input('Would you like to visualise the distributions and QQ plots of the successfully' 
                                   ' transformed features? Y/N ')
        if 'y' in visualise_dist.lower():
            visualise = True
        else: 
            visualise = False
            
        for feature, skew_score in highly_skewed.items():
            transformation = {feature : {}}
            #print(f'Transforming {feature} - initial skew score: {skew_score:.4f}')
            if feature in data.columns :
                #print(f'{feature} found - ready to transform')
                original_data = data[feature]
                yeo_johnson_transformed, lambda_val = yeojohnson(original_data)
                    
                transformation[feature]['Untransformed'] = original_data
                transformation[feature]['Square Root'] = np.sqrt(original_data)
                transformation[feature]['Cube Root'] =np.cbrt(original_data)
                transformation[feature]['Fourth Root'] = np.power(original_data, 1/4)
                transformation[feature]['Fifth Root'] = np.power(original_data, 1/5)
                transformation[feature]['Log plus 1'] = np.log1p(original_data)
                transformation[feature]['Yeo-Johnson'] = yeo_johnson_transformed
                transformation[feature]['Arcsine'] = np.arcsin(np.sqrt(original_data))
                transformation[feature]['Exponential'] = np.exp(original_data)
                transformation[feature]['Inverse'] = np.reciprocal(original_data)
                
                
                def compare_transforms(feature, transformations, alpha, visualise):
                    alpha100 = alpha*100
                    print('\033[1m'+f'\nComparing Transformations for {feature} at {alpha100}% significance level\n'+ '\033[0m')
                    
                    stat_result = {}
                    #successful_stat_result = {}
                    
                    result_list = []
                    #successful_result_list = []
                    
                    transform_list = []
                    
                    for feature, transform_data in transformations.items():
                        for transform, data in transform_data.items():
                            stat_result = {transform: {}}
                        
                        #statistical tests
                            shapiro_statistic, shapiro_p_value = shapiro(data)
                            ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(data, 'norm')
                            ks_statistic, ks_p_value = stats.kstest(data, 'norm')
                            feat_skew = skew(data)
                        
                        #storing results
                            stat_result[transform]['shapiro_statistic'] = shapiro_statistic
                            stat_result[transform]['shapiro_p_value'] = shapiro_p_value
                            
                            title = f'{feature} after {transform}'
                            
                            
                            if shapiro_p_value < alpha:
                                stat_result[transform]['shapiro_result'] = 'Not Normally Distributed'
                            elif str(shapiro_statistic) == 'nan':
                                stat_result[transform]['shapiro_result'] = 'Test Failed'
                            else:
                                stat_result[transform]['shapiro_result'] = 'Normally Distributed'
                            
                            stat_result[transform]['ks_statistic'] = ks_statistic
                            stat_result[transform]['ks_p_value'] = ks_p_value
                            
                            if ks_p_value < alpha:
                                 stat_result[transform]['ks_result'] = 'Not Normally Distributed'
                            elif str(ks_p_value) == 'nan' or str(ks_statistic) == 'nan':
                                stat_result[transform]['ks_result'] = 'Test Failed'
                            else:
                                stat_result[transform]['ks_result'] = 'Normally Distributed'     

                
#ad test has multiple crit values and which to use depends on alpha. so storing the 
#result is slightly more complex but can find alpha within list of significance levels,
#then use the index of alpha within the list to find the correct critical value

                        #making significance levels uniform to allow indexing    
                            sig_levels = []
                            for sf in ad_significance_levels:
                                sf = float(sf)
                                sig_levels.append(sf)
                        
                        #creating index and finding alpha within sig_levels
                            sig_index = None  
                            for i, value in enumerate(sig_levels):
                                if value == alpha100:
                                    sig_index = i
                        
                        #storing result at that significance level for ad test
                            stat_result[transform][f'ad_critical_value'] = ad_critical_values[sig_index]
                            stat_result[transform][f'ad_statistic'] = ad_statistic
                            ad_critical_value = ad_critical_values[sig_index]
                            
                            if ad_statistic < ad_critical_value:
                                stat_result[transform]['ad_result'] = 'Normally Distributed'
                            elif str(ad_statistic) == 'nan':
                                stat_result[transform]['ad_result'] = 'Test Failed'
                            else:
                                stat_result[transform]['ad_result'] = 'Not Normally Distributed'
                                
                                
                            if str(feat_skew) == 'nan':
                                stat_result[transform]['skew'] = 'Test Failed'
                            else:
                                stat_result[transform]['skew'] = feat_skew
                            
                            result_list.append(stat_result)
                           
                            
                            if (stat_result[transform]['shapiro_result'] or stat_result[transform]['ks_result'] or stat_result[transform]['ad_result']) == 'Normally Distributed':
                                test_results = [stat_result[transform]['shapiro_result'], stat_result[transform]['ks_result'], stat_result[transform]['ad_result']]
                                transform_results= stat_result.keys()
                                test_names = ['Shapiro test', 'KS test', 'AD test']
                                success_list = []
                                
                                
                                for index, (name, result) in enumerate(zip(test_names, test_results), start=1):
                                    if result == 'Normally Distributed':
                                        success_list.append(name)
                                        transform_list.append(transform_results)
                                if visualise:
                                    plot_title = f'{feature} after {transform}'
                                    self.plot_skewness(data, plot_title)
                                    self.QQ_plot(data, plot_title)
                                    print(f'{feature} normally distributed based on {success_list}\n')

                    df = pd.DataFrame.from_dict({list(d.keys())[0]: list(d.values())[0] for d in result_list}, orient='index')
                    print(df)
                    
                
                    if transform_list:
                        while True:
                            transform_choice = input(f'\nWhich transformation would you like to apply to {feature} out of {transform_list}? ')
                            transform_choice = transform_choice.lower().title()
                            try:
                                self.data[feature] = transformations[feature][transform_choice]
                                break
                            except KeyError:
                                print(f'{transform_choice} is an invalid selection')
                    
                    shapiro_success = df['shapiro_result'].value_counts().get('Normally Distributed', 0)
                    ks_success = df['ks_result'].value_counts().get('Normally Distributed', 0)
                    ad_success = df['ad_result'].value_counts().get('Normally Distributed', 0)

                    
                    if shapiro_success == 0:
                        print('\nThe Shapiro-Wilk test is sensitive to small sample sizes but may' 
                              ' lack accuracy with non-normal distributions.')
                        
                    if ks_success == 0:
                        print('\nKolmogorov-Smirnov test is versatile but less powerful' 
                              ' for detecting deviations from normality.')
                              
                    if ad_success == 0:
                        print('\nAnderson-Darling test is effective for various sample'
                              ' sizes but might be influenced by extreme values.')
                        if self.outliers:
                            print('Consider removing outliers and retesting')
                    
                    
                        
                        #outside the inner for loop
                    #outside the outer for loop
                    
                    
            compare_transforms(feature, transformation, alpha, visualise)
        return self.data
        

                
    def plot_skewness(self, data, title):
        try:
            sns.histplot(data, bins=50, alpha=0.7, kde = True)
            plt.title(f'Distribution of {title}')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            title = title.replace(' ', '_')
            file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/class_dist/histograms/{title}'
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            print(f"Skewness: {skew(data)}\n")
        except ValueError:
            print(f'{title} unable to be graph - may not be finite')
            
    
    def QQ_plot(self, data, title):
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {title}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/class_dist/qq_plots/{title}'
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight') 
        plt.show()
        
        
    
    def outlier_removal(self, mad = False):
        
        
        #going to give option for DBSCAN or HDBSCAN 
        
        def k_nearest_plot(data, z_score = True):
            np.set_printoptions(precision=4)
            
            if data is self.data:
                title = 'with outliers retained'
            elif z_score:
                title = 'based on Z-Score'
            else: 
                title = 'based on MAD'
                
        
            if data is None:
                raise ValueError('Dataframe does not exist')
            else:
                print('\033[1m' +f'\nEstimating min_samples using K-Nearest Neighbour plots {title}\n'+ '\033[0m')
            
            
            num_dimensions = len(list(data.columns))
            quart_rows = int(round(data.shape[0] * 0.25, 0))
            k_values = range(num_dimensions, quart_rows)
            
            avg_distances = []
            distances_list = []
            distances_dict = {}
            avg_distances_dict = {}
            reach_dict = {}
            
            
            normalised_data = data.copy()
            normalised_data=RobustScaler().fit_transform(normalised_data)
            
            for k in k_values:
                nearest_neighbors = NearestNeighbors(n_neighbors=k + 1)
                nearest_neighbors.fit(normalised_data)  
                distances, _ = nearest_neighbors.kneighbors(normalised_data)
                avg_distances.append(np.mean(distances[:, -1]))
                distances_list.append(distances[:, -1])
                distances_dict[k] = distances
                avg_distances_dict[k] = np.mean(distances)
                reach = np.max(distances[:, 1:], axis=1)
                reach_dict[k] = np.sort(reach)

            
            
            distances_array = np.array(distances_list)
            avg_distances_array = np.array(avg_distances)
            avg_dist_diffs = np.diff(avg_distances_array)
            #dist_diffs = np.diff(distances_array)
            
            avg_elbow_index = np.argmax(avg_dist_diffs)
            #max_index = np.argmax(dist_diffs)
            
            avg_elbow_array_index = np.where(avg_dist_diffs == np.max(avg_dist_diffs))

            
            #plus one to overcome 0 indexing
            #max_dist_diff = int(avg_elbow_array_index[0]) + 1
            max_dist_diff = int(avg_elbow_array_index[0]) + 1 + num_dimensions
            
            
            def define_threshold(dist_diffs):
                while True:
                    try:
                        threshold_factor = float(input('How many standard deviations away from the mean should a data point be to be estimated as part of a new cluster? '))
                        break 
                    except ValueError:
                        print("Invalid input - please enter a numeric value.")
            
                threshold = np.mean(dist_diffs) + threshold_factor * np.std(dist_diffs)
                non_zero_diffs = dist_diffs[dist_diffs != 0]
                non_zero = np.mean(non_zero_diffs) + threshold_factor *np.std(non_zero_diffs)
                
                if threshold == non_zero:
                    print(f'\nThreshold: {threshold}')
                    return threshold
                else:
                    print(f'\nThreshold: {non_zero}')
                    print('Zeros in difference array')
                    return non_zero
                
            
            potential_knee_points = []
            k_near_threshold = []
            
            
            while True:
                if len(potential_knee_points) == 0:
                    threshold = define_threshold(avg_dist_diffs)
                    for idx, value in enumerate(avg_dist_diffs):
                        if value >= threshold:
                            potential_knee_points.append(idx+1+num_dimensions)
                        elif value < threshold and value > 0.9*threshold:
                            k_near_threshold.append(idx+1+num_dimensions)
                            
                    if len(potential_knee_points) == 0:
                        print('Sigma value too high - no knee points identified')
                    else:
                        print('Potential Elbow points found')
                else:
                    break
                    
            k_ranges_dict = {}
            knee_ranges = []
            
            if len(potential_knee_points) > 1:
                print('\nCalculating ranges')
                for i in range(len(potential_knee_points)):
                    knee_ranges.append(potential_knee_points[i] - potential_knee_points[0])
                    
                knee_range_diffs = np.diff(knee_ranges)
                
                idx = 0
                
                while idx < len(knee_range_diffs):
                    values = knee_range_diffs[idx]
                
                    if abs(int(values)) > 10 and abs(int(values))<80:
                        if not k_ranges_dict:
                            if potential_knee_points[0] < 11:
                                min_k = 1
                            else:
                                min_k = potential_knee_points[0] - 10
                            max_k = potential_knee_points[-1] + 10
                            k_range = range(min_k, max_k)
                        else:
                            keys = list(k_ranges_dict.keys())
                            last_key = keys[-1]
                            min_k = k_ranges_dict[last_key].start
                            max_k = potential_knee_points[idx+1] + 11
                            k_range = range(min_k, max_k)
                            k_ranges_dict[last_key] = k_range                        
                        idx += 1
                                
                            
                        
                    elif abs(int(values)) >= 80:
                        if potential_knee_points[idx] < 11:
                            p_min_k = 1
                        else:
                            p_min_k = potential_knee_points[idx] - 10
                        p_max_k = potential_knee_points[idx] + 11
                        min_k = potential_knee_points[idx+1] - 10
                        max_k = potential_knee_points[idx+1] + 11
                        p_k_range = range(p_min_k, p_max_k)
                        c_k_range = range(min_k, max_k)
                        p_k = potential_knee_points[idx]
                        c_k = potential_knee_points[idx+1]
                        
                        k_ranges_dict[p_k] = p_k_range
                        k_ranges_dict[c_k] = c_k_range
                                
                        idx += 1
            
                    else:
                        if not k_ranges_dict:
                            if potential_knee_points[0] < 11:
                                min_k = 1
                            else:
                                min_k = potential_knee_points[0] - 10
                            max_k = potential_knee_points[i] + 11
                            k_range = range(min_k, max_k)
                        else:
                            keys = list(k_ranges_dict.keys())
                            last_key = keys[-1]
                            min_k = k_ranges_dict[last_key].start
                            max_k = potential_knee_points[idx+1] + 11
                            k_range = range(min_k, max_k)
                            k_ranges_dict[last_key] = k_range        
                        
                        idx += 1 
                                
                    
            else: 
                if potential_knee_points[0] < 11:
                    min_k = 1
                else:
                    min_k = potential_knee_points[0] - 10
                max_k = potential_knee_points[0] + 11
                k_range = range(min_k, max_k)
                
                
            k_to_plot = []
            distances_to_plot=[]
            if k_ranges_dict:
                print('Multiple plots required')
                print(f'Ranges that need to be plotted {k_ranges_dict}')
                for ks, k_range in k_ranges_dict.items():
                    num_points = len(list(k_range))
                    figsize = (min(10 + num_points * 0.2, 18), min(8 + num_points * 0.15, 12))
                    plt.figure(figsize=figsize)
                    for k in k_range:
                        if k in avg_distances_dict:
                            distances = avg_distances_dict[k]
                            distances_to_plot.append(distances)
                            k_to_plot.append(k)
                   
                    plt.plot(k_to_plot, distances_to_plot, marker ='o', color = 'black', linestyle = '-')
                    
                    for potential_knee_point in potential_knee_points:
                        if potential_knee_point in avg_distances_dict:
                            distances = avg_distances_dict[potential_knee_point]
                            plt.scatter(potential_knee_point, distances, color='red', s=100)
                            
                    plt.xlabel('Number of Neighbors (k)')
                    plt.ylabel('Distance to k-th Nearest Neighbor')
                    plt.title(f'K-Distance Plot for neighbours {title}')
                    file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/dbscan/min_samples/{k_range}/{title}'
                    directory = os.path.dirname(file_path)
                    os.makedirs(directory, exist_ok=True)
                    plt.savefig(file_path, bbox_inches='tight') 
                    plt.show()
                            
           
            else:
                num_points = len(list(k_range))
                figsize = (min(10 + num_points * 0.2, 18), min(8 + num_points * 0.15, 12))
                plt.figure(figsize=figsize)
                print('Range can be plotted on single graph')
                print(f'K Range: {k_range}')
                
            
                for k in k_range:
                    if k in avg_distances_dict:
                        distances = avg_distances_dict[k]
                        distances_to_plot.append(distances)
                        k_to_plot.append(k)

                plt.plot(k_to_plot, distances_to_plot, marker ='o', color = 'black', linestyle = '-', markersize = 4)
                
                for potential_knee_point in potential_knee_points:
                    if potential_knee_point in avg_distances_dict:
                        distances = avg_distances_dict[potential_knee_point]
                        plt.scatter(potential_knee_point, distances, color='red', s=100)
                        
                plt.xlabel('Number of Neighbors (k)')
                plt.ylabel('Distance to k-th Nearest Neighbor')
                plt.title(f'K-Distance Plot for neighbours {title}')
                file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/dbscan/min_samples/{k_range}/{title}'
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)
                plt.savefig(file_path, bbox_inches='tight') 
                plt.show()
            
            
            num_points = len(list(k_values))
            figsize = (min(10 + num_points * 0.2, 18), min(8 + num_points * 0.15, 12))
            plt.figure(figsize=figsize)
            distances = avg_distances_dict.values()
            k = avg_distances_dict.keys()
            plt.plot(k, distances, marker = 'o', color = 'black', linestyle='-', markersize = 4)
                        
            for potential_knee_point in potential_knee_points:
                if potential_knee_point in avg_distances_dict:
                    distances = avg_distances_dict[potential_knee_point]
                    plt.scatter(potential_knee_point, distances, color='red', s=100)
                
            for near_threshold in k_near_threshold:
                distances = avg_distances_dict[near_threshold]
                plt.scatter(near_threshold, distances, color = 'green', s=100)
                
            plt.xlabel('Number of Neighbors (k)')
            plt.ylabel('Distance to k-th Nearest Neighbor')
            plt.title(f'Full K-Distance Plot for neighbours {title}')
            file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/dbscan/min_samples/full_range/{title}'
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            plt.savefig(file_path, bbox_inches='tight') 
            plt.show()
                
           
            print(f'Largest difference in average distance K = {max_dist_diff}')
            print(f'Potential elbow points K = {potential_knee_points}') 
            if len(k_near_threshold) != 0:
                print(f'K within 90% of threshold = {k_near_threshold}\n')
            else:
                print('No other points were close to the threshold\n')
                
            
            def plot_reachability(reach_dict, knee_points, data_name = 'outliers_retained'):
                print('\033[1m' +'\nPlotting Reachability to Estimate Epsilon'+ '\033[0m')
                print('In both K-distance and reachability look for sudden'
                      ' jumps or elbows.\n')
            
                k_list = []
                
                for knee_point in knee_points:
                    k_list.append(knee_point-1)
                
                while True:
                    select_another = input('Would you like to analyse another K? Y/N ')
                    if 'y' in select_another.lower():
                        while True:
                            try:
                                optimum_k = int(input('Based off the above graphs at what K does the value increase most rapidly? '))
                                optimum_k = optimum_k - 1
                                k_list.append(optimum_k)
                                print(f'K List: {k_list}')
                                break
                            except ValueError:
                                print(f'{optimum_k} is an invalid input - please enter an integer')
                    elif 'n' in select_another.lower():
                        break
                    else:
                        print(f'{select_another} is invalid please enter Y/N')
                    
                
                while True:
                        try:
                            sample = float(input('What percentage of the dataset would you like to use in calculating local statstical metrics '))
                            break
                        except ValueError:
                            print('Invalid sample size')
                    
                    
                while True:
                        try:
                            threshold_factor = float(input('\nHow many standard deviations from the local mean should be considered as significant? '))
                            break
                        except ValueError:
                            print(f'{threshold_factor} is invalid please enter a number')
                
                init_size = self.data.shape[0]
                sample_size = self.data.shape[0]*(sample/100)
                    
                sample_size = int(round(sample_size, 0))
                
                eps_list = []
                selected_eps = set()
                
                for optimum_k in k_list:

                    epsilon_list = list(reach_dict[optimum_k])
                    reach_diffs = np.diff(reach_dict[optimum_k])

                    reach_diffs_index = {}
                    for i, value in enumerate(reach_diffs):
                        reach_diffs_index[value] = i+1

                    local_avg = []
                    local_std = []
                    
                    #if isinstance(sample_size, int):
                     #   for i in range(len(reach_diffs) - sample_size + 1):
                      #      sample_values = reach_diffs[i:i+sample_size]
                       #     local_avg.append(np.mean(sample_values))
                        #    local_std.append(np.std(sample_values))
                            
                    if isinstance(sample_size, int):
                        for i, value in enumerate(reach_diffs):
                            if (i - sample_size/2) < 0:
                                x = (i - sample_size/2)
                                a = (sample_size/2) + x
                                b = (sample_size/2) - x
                            elif i + sample_size/2 > len(reach_diffs):
                                b = (sample_size/2)-((i + sample_size/2)-len(reach_diffs))
                                a = sample_size/2 + ((i + sample_size/2)-len(reach_diffs))
                            else:
                                a = sample_size/2
                                b = sample_size/2
                                
                                a_string = str(a)
                                if '.5' in a_string:
                                    if i - (a + 0.5) < 0:
                                        a = a - 0.5
                                        b = b +0.5
                                    elif i + (b+0.5) > len(reach_diffs):
                                        a = a+0.5
                                        b = b-0.5
                                    else:
                                        if i <= len(reach_diffs)/2:
                                            a = a + 0.5
                                            b = b - 0.5
                                        else:
                                            a = a - 0.5
                                            b = b+0.5
                                            
                                    
                            a= int(round(a, 0))
                            b = int(round(b, 0))
                            sample_values = reach_diffs[i-a:i+b]
                            local_avg.append(np.mean(sample_values))
                            local_std.append(np.std(sample_values))
                        
                    else:
                        print('Invalid sample size')
                    
                    print(f'\nInitial Dataframe size: {init_size}')
                    print(f'Sample Size: {sample_size}')
                    print('Number of Local Averages ' + str(len(local_avg)))
                    print('Length of Difference in Reachability ' +str(len(reach_diffs)))
                    
                    over_threshold={}
                    for i, (avg, std) in enumerate(zip(local_avg, local_std)):
                        threshold = avg + threshold_factor * std
                        if reach_diffs[i] >= threshold:
                            value = epsilon_list[i]
                            over_threshold[i] = value
                            eps_list.append(value)
                            selected_eps.add(round(value, 2))
                        
                    sorted_diffs = np.sort(reach_diffs)[::-1]
                    largest_diffs = sorted_diffs[0:20]
                
                    num_points = len(list(reach_dict[optimum_k]))
                    figsize = (min(10 + num_points * 0.2, 18), min(8 + num_points * 0.15, 12))
                    plt.figure(figsize=figsize)
            
                    plt.plot(reach_dict[optimum_k], marker='o', linestyle='-', color='blue', markersize=1)
                
                    print('\nLikely Elbow Points:')
                    print('Red represents the 20 largest changes in distance')
                    print('Green represents points that are over the set threshold\n')
                    index_list = []
                    eps_index = {}
                    
                    for diffs in list(largest_diffs):
                        index = reach_diffs_index[diffs]
                        epsilon_index = epsilon_list[index]
                        eps_index[index] = epsilon_index
                        index_list.append(index)
                        
                    
                    sorted_keys = sorted(eps_index.keys())
                    sorted_dict_by_keys = {key : eps_index[key] for key in sorted_keys}
                
                    for index, epsilon_value in sorted_dict_by_keys.items():
                        plt.scatter(index, epsilon_value, color='red', s=100)
                        eps_list.append(epsilon_value)
                        
                        
                    for index, epsilon_value in over_threshold.items():
                        plt.scatter(index, epsilon_value, color = 'green', s=100)
                    

                
                    plt.xlabel('Point Index')
                    plt.ylabel('Reachability Distance')
                    plt.title(f'Reachability Plot for DBSCAN Epsilon Estimation for {optimum_k} neighbours')
                    file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/dbscan/eps/{data_name}/{optimum_k}_neighbours'
                    directory = os.path.dirname(file_path)
                    os.makedirs(directory, exist_ok=True)
                    plt.savefig(file_path, bbox_inches='tight') 
                    plt.show()
                
                    index_list = np.sort(index_list)

                min_samples = []
                for k in k_list:
                    min_samples.append(k+1)

                eps_to_remove = []
                for i, eps in enumerate(eps_list):
                    if round(eps, 2) not in selected_eps:
                        eps_to_remove.append(i)
                
                eps_to_remove.sort(reverse = True)
                
                for index in eps_to_remove:
                    del eps_list[index]
                    
                results = {'eps': eps_list, 'min_samples': knee_points}
                
                return results
            
            if data is self.data:
                params = plot_reachability(reach_dict, potential_knee_points)
                return params
            elif data is self.outliers_removed:
                z_params = plot_reachability(reach_dict, potential_knee_points, 'z_score_rem')
                return z_params
            else:
                mad_params = plot_reachability(reach_dict, potential_knee_points, 'MAD_rem')
                return mad_params
        
        if self.outliers_removed is not None or self.mad_outliers_removed is not None:
            if not mad:
                #try:
                z_params = k_nearest_plot(self.outliers_removed)
                z_min_samples = z_params['min_samples']
                z_eps = z_params['eps']

                    
                if len(z_min_samples) and len(z_eps) != 0:
                    print('\nParameters Selected')
                    if not self.stop_cluster:
                        print('\033[1m' +'\nBeginning DBSCAN\n'+ '\033[0m')
                        self.dbscan_params = z_params
                        self.dbscan(z_params)   
                    else:
                        print(f'Stop cluster is {self.stop_cluster}')
                else:
                    print('\nParameters not saved')
                    
            elif mad:
                try:
                    mad_params = k_nearest_plot(self.mad_outliers_removed, z_score = False)
                    mad_min_samples = mad_params['min_samples']
                    mad_eps = mad_params['eps']

                    if len(mad_min_samples) and len(mad_eps) != 0:
                        self.stop_cluster = False
                        print('\nParameters Selected')
                        if not self.stop_cluster:
                            print('\033[1m' +'\nBeginning DBSCAN\n'+ '\033[0m')
                            self.dbscan_params = mad_params
                            self.dbscan(mad_params)
                        else:
                            print(f'Stop cluster is {self.stop_cluster}')
                    else:
                        print('\nParameters not saved')
                except ValueError:
                    print('No outliers removed by MAD method')

                    
        else:
            params = k_nearest_plot(self.data)
            min_samples = params['min_samples']
            eps = params['eps']
           

            if len(min_samples) and len(eps) != 0:
                print('\nParameters Selected')
                self.dbscan_params = params
                self.dbscan(params)
            else:
                print('Parameters not saved')
                
        
    def dbscan(self, params, init_silhouette_threshold = 0.2, silhouette_threshold = None, eps_threshold = None):
        
        
        #self.stop_cluster = False
        warnings.simplefilter("ignore")
        data = self.data
        
        if not self.silhouette_threshold:
            silhouette_threshold = init_silhouette_threshold
        else:
            silhouette_threshold = round(self.silhouette_threshold, 1)
        
        
        if not silhouette_threshold:
            silhouette_threshold = 0
        else:
            pass
                
            
        if self.encoded_features:
            data = gower_matrix(data)
            title = 'based on Gower Distance'
            dbscan_metric = DBSCAN(metric = 'precomputed')
        else:
            data = data
            title = 'based on default metrics'
            dbscan_metric = DBSCAN()            
            
        def silhouette_scorer(estimator, X):
            labels = estimator.fit_predict(X)
            return silhouette_score(X, labels)
            
        eps_range = params['eps']
        min_samples_range = params['min_samples']
        init_params = params
        dbscan_results = {}
        attempted_eps = []
        
        #if init_params is self.dbscan_params:
         #   print('Initial Parameters Saved')
        #else: 
         #   print('Initial Parameters not saved')
            
        silhouette_scores = {}
        silhouette_score2= {}
            
        data_points =self.data.shape[0]*self.data.shape[1] 

        if not eps_threshold:
            if data_points < 500:
                threshold = 300
            else:
                fractional_multiplier = (len(eps_range) * len(min_samples_range)) / data_points
                target_product = range(150, 300)  
                threshold = round(data_points/ (len(eps_range) + len(min_samples_range)))
                while True:

                #if fractional_multiplier < target_product / data_points:
                    if threshold < target_product.start:
                        #print(f'Increasing Threshold - Current: {threshold}')
                    
                        if fractional_multiplier < 1:
                            threshold = threshold/fractional_multiplier
                        else: 
                            threshold = threshold*fractional_multiplier  
                        #print(f'New threshold: {threshold}')
                        fractional_multiplier =  fractional_multiplier / (data_points/(target_product.start + len(eps_range)+len(min_samples_range)))
                        #print(f'Multiplier: {fractional_multiplier}\n')

                    elif threshold > target_product.stop:
                        #print(f'Reducing Threshold: {threshold}')
                        if fractional_multiplier > 1:
                            threshold = threshold/fractional_multiplier
                        else: 
                            threshold = threshold*fractional_multiplier
                        #print(f'New threshold: {threshold}')
                        fractional_multiplier = fractional_multiplier*(len(eps_range)+len(min_samples_range))
                        #print(f'Multiplier: {fractional_multiplier}\n')
                    
                
                    else:
                        #coeff = data_points / fractional_multiplier
                        #print(f'Threshold Suitable: {threshold}')
                        break

        else: 
            threshold = eps_threshold
            
        #Use the calculated coeff for further computations or clustering
# For instance:
        #threshold = (data_points * coeff) - (len(eps_range) / len(min_samples_range))
        threshold = round(threshold)
        
        while not self.stop_cluster:
            
            iterr = 1
            if iterr == 1:
                print(f'Threshold: {threshold}')
            else:
                pass

            while True:
                #threshold = (data_points/coeff) 
                #old_eps_range = len(eps_range)
                if len(eps_range) > threshold:    
                    #if len(eps_range) >= 1000:
                     #   n_iters = int(round(len(eps_range)/(2*len(min_samples_range)), 0))
                    #elif len(min_samples_range) < 5 and len(eps_range) > 500 and len(eps_range) < 1000:
                     #   n_iters = int(round(len(eps_range)/10))
                        
                    num_eps = float(len(eps_range))
                    num_samples = float(len(min_samples_range))
                    if num_eps > 100:
                        multiplier = 0.001
                    else:
                        multiplier = 0.01
                    
                    n_iters = 50 + 150*10**-(multiplier*num_eps+0.1*num_samples)
                    n_iters = int(round(n_iters))
                        
                    print('\033[1m' + '\nReducing Number of Epsilon Values' +'\033[0m')
                    print(f'Iteration: {iterr}')
                    print(f'Threshold: {threshold:.0f}')
                    print(f'Current Number of Epsilon: {len(eps_range)}')
                    print(f'Number of min_samples: {len(min_samples_range)}')
                    print(f'Number of combination: {n_iters}')
                    print('\n Progress')
                    iterr += 1
                    new_eps_range = []
                    for i, min_sample in enumerate(min_samples_range):
                    # Perform RandomizedSearchCV to determine a new range of eps
                        param_grid = {'eps': eps_range} 
                        dbscan = DBSCAN(min_samples=min_sample)
                        
                        scorer = make_scorer(silhouette_scorer)
                        
                        random_search = RandomizedSearchCV(dbscan, param_distributions=param_grid, 
                                                           n_iter=n_iters, cv=5, scoring = scorer)
                        random_search.fit(data)

                        best_eps = random_search.best_params_['eps']                        
                        n = int(round(threshold/(1.5*len(min_samples_range)), 0)) - 1
                        
                      
                        #n = int(round(2*threshold/len(min_samples_range)) - (len(eps_range)/len(min_samples_range))) + 1
                        
                        top_eps_values = [params['eps'] for params in random_search.cv_results_['params'][:n]]
                        
                        new_eps_range.append(top_eps_values)
                        progress = round(((i+1)/(len(min_samples_range)+1))*100, 1)
                        print(f'{progress}%')
                        
                        
                    eps_range = [eps for eps_ranges in new_eps_range for eps in eps_ranges]
                    print(f'Updated Number of Epsilon {len(eps_range)}')
                    
                else:
                    print('\033[1m' + f'\nAttempting to cluster {title}' +'\033[0m')
                    print(f'Silhouette Threshold: {silhouette_threshold}')
                    print(f'Number of Epsilons: {len(eps_range)}\n')
                    break
         
            #dbscan_results = {}
            iteration = 1

        
            #while True:
            print(f'stop cluster is {self.stop_cluster}')
            #threshold = threshold  + (len(min_samples_range)/len(eps_range))
            #threshold = round(threshold)
                                                
            #    if not dbscan_results:
            while not dbscan_results and not self.stop_cluster:
                print(f'\nAttempting to cluster, iteration: {iteration}')
                print(f'Threshold: {threshold}')

                    
                
                new_eps_range = []
                rounded_eps = []
                for eps in eps_range:
                    rounded_eps.append(round(eps, 2))
                    attempted_eps.append(eps)
                    for min_samples in min_samples_range:
                        param = (eps, min_samples)
                        try:
                            if self.encoded_features:
                                dbscan_model = DBSCAN(eps = eps, min_samples=min_samples, metric = 'precomputed')
                            else:
                                dbscan_model = DBSCAN(eps = eps, min_samples=min_samples)
                            
                            labels = dbscan_model.fit_predict(data)
                            silhouette = silhouette_score(self.data, labels)
                            silhouette_scores[param] = silhouette
                        
                        # Core samples are plotted as filled in
                            core_samples_mask = np.zeros_like(labels, dtype=bool)
                            core_samples_mask[dbscan_model.core_sample_indices_] = True

                        # Number of clusters in labels, ignoring noise if present.
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            n_noise_ = list(labels).count(-1)
                            unique_labels = set(labels)
                            
                            if 1 in unique_labels and silhouette > silhouette_threshold:
                                if n_clusters_ not in dbscan_results:
                                    dbscan_results[n_clusters_] = {'silhouette':silhouette, 'eps': eps,
                                                                   'min_samples':min_samples, 'labels':labels,
                                                                   'model':dbscan_model}
                                elif n_clusters_ in dbscan_results and silhouette > dbscan_results[n_clusters_]['silhouette']:
                                    dbscan_results[n_clusters_] = {'silhouette':silhouette, 'eps': eps,
                                                                   'min_samples':min_samples, 'labels':labels,
                                                                  'model':dbscan_model}
                            else:
                                pass
                                
                        except ValueError:
                            div_eps = eps/2
                            multi_eps = eps*1.8
                            if round(div_eps,2) not in rounded_eps:
                                new_eps_range.append(div_eps)
                            else:
                                pass
                            if round(multi_eps,2) not in rounded_eps:
                                new_eps_range.append(multi_eps)
                            else:
                                 pass
                            
                
                def approximately_equal(value1, value2):
                    return round(value1, 2) == round(value2, 2)

                def remove_similar_values(values, attempted_values):
                    i = 0
                    while i < len(values):
                        j = i + 1
                        while j < len(values):
                            if approximately_equal(values[i], values[j]):
                                del values[j] 
                            else:
                                j += 1
                        i += 1
                        
                    values = [value for value in values if not any(approximately_equal(value, attempted_value) for attempted_value in attempted_values)]
                    return values
                    
                    
                print(f'Previous number of eps: {len(eps_range)}')
                eps_range = remove_similar_values(new_eps_range, attempted_eps)
                print(f'Current number of eps: {len(eps_range)}\n')
               
                    
                if iteration % 5 == 0:
                   # min_eps = min(new_eps_range)
                   # max_eps = max(new_eps_range)
                   # print(f'Current eps in range {min_eps}-{max_eps}')
                   # silhouette_threshold = silhouette_threshold - 0.1
                   # self.silhouette_threshold = silhouette_threshold
                   # self.dbscan(self.dbscan_params, init_silhouette_threshold= None, silhouette_threshold = silhouette_threshold, eps_threshold = threshold)
                    
                    if silhouette_threshold == 0:
                        while True:
                            cont = input('Would you like to continue trying to cluster? Y/N ')
                            if 'y' in cont.lower():
                                silhouette_threshold = silhouette_threshold - 0.1
                                self.silhouette_threshold = silhouette_threshold
                                self.dbscan(self.dbscan_params, init_silhouette_threshold= None, silhouette_threshold = silhouette_threshold, eps_threshold = threshold)
                                break
                            elif 'n' in cont.lower():
                                self.stop_cluster = True
                                print(self.stop_cluster)
                                break
                            else:
                                print(f'{cont} is an invalid input please enter Y/N')
                            
                    elif silhouette_threshold < -0.3:
                        self.stop_cluster = True
                
                    else:
                        pass 
                    
                    min_eps = min(new_eps_range)
                    max_eps = max(new_eps_range)
                    print(f'Current eps in range {min_eps}-{max_eps}')
                    silhouette_threshold = silhouette_threshold - 0.1
                    self.silhouette_threshold = silhouette_threshold
                    self.dbscan(self.dbscan_params, init_silhouette_threshold= None, silhouette_threshold = silhouette_threshold, eps_threshold = threshold)
                    
                
                
                if len(eps_range) > threshold:
                    print('Number of epsilon exceeded threshold')
                    new_samples = []
                    for min_sample in min_samples_range:
                        new_samples.append(min_sample)
                    params = {'eps': eps_range, 'min_samples':new_samples}
                    if not self.silhouette_threshold:
                        return self.dbscan(params, eps_threshold = threshold)
                    else:
                        return self.dbscan(params, silhouette_threshold = self.silhouette_threshold, eps_threshold = threshold)
                else:
                    iteration += 1
                    
 
            if dbscan_results:
            
                print('DBSCAN clustered successfully')
                self.stop_cluster = True
                
                               
      
                for n_clusters_, results in dbscan_results.items():
                    labels = dbscan_results[n_clusters_]['labels']
                    model = dbscan_results[n_clusters_]['model']
                    core_samples_mask = np.zeros_like(labels, dtype=bool)
                    core_samples_mask[model.core_sample_indices_] = True
                    n_noise_ = list(labels).count(-1)
                    unique_labels = set(labels)
                    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                
                    np.random.seed(42)
                    random_labels = np.random.randint(0, n_clusters_, size=len(data))
                    random_silhouette = silhouette_score(data, random_labels)
                    
                    plt.figure(figsize=(15, 10))

                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # Black used for noise.
                            col = [0, 0, 0, 1]
                        
                        class_member_mask = (labels == k)

                        xy = data[class_member_mask & core_samples_mask]
                        xy = pd.DataFrame(xy)
                        plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1], s=50, c=[col], marker='o', edgecolor='k')
                            #plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker='o', edgecolor='k')

                        xy = data[class_member_mask & ~core_samples_mask]
                        xy = pd.DataFrame(xy)
                        plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1], s=50, c=[col], marker='x', edgecolor='k')
                            #plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker='x', edgecolor='k')

                    eps = results['eps']
                    min_samples = results['min_samples']
                    rounded_param = (round(eps,2), min_samples)
                    plt.title(f'DBSCAN{rounded_param} {title}')
                    plt.xlabel('X-axis')
                    plt.ylabel('Y-axis')
                    file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/dbscan/cluster/{n_clusters_}_clusters/DBSCAN{rounded_param}_{title}.png'
                    directory = os.path.dirname(file_path)
                    os.makedirs(directory, exist_ok=True)
                    plt.savefig(file_path, bbox_inches='tight') 
                    plt.show()
                        
                    tsne = TSNE(n_components=2, random_state=42)
                    data_tsne = tsne.fit_transform(data)
                        
                    plt.figure(figsize=(15, 10))
                    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c= labels, cmap='viridis', s=50)
                    plt.title(f'DBSCAN{rounded_param} {title} after TSNE dimension reduction')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                    #plt.colorbar()
                    file_path = f'{self.home}/Desktop/Coding/projects/{self.project_name}/figures/dbscan/cluster/{n_clusters_}_clusters/dim_red_DBSCAN{rounded_param}_{title}.png'
                    directory = os.path.dirname(file_path)
                    os.makedirs(directory, exist_ok=True)
                    plt.savefig(file_path, bbox_inches='tight') 
                    plt.show()
                    silhouette = dbscan_results[n_clusters_]['silhouette']
                    print(f'DBSCAN Silhouette Score: {silhouette}')
                    print(f'Random Silhouette Score: {random_silhouette}')
            
            
                #break
                
                #elif self.stop_cluster:
                 #   print('Exiting')
                  #  break
                    
                #else:
                  #  print('Something has gone wrong DBSCAN is None and not None simultaneously')
                  #  pass

        

                
            
        
        def HDBSCAN_outliers(data):
            pass
        
        
       # while True:
        #    clustered = input('Are you satisfied with the DBSCAN clustering? Y/N ')
         #   if 'y' in clustered.lower():
          #      break
           # elif 'n' in clustered.lower():
            #    hdbcan(data)
             #   break
            #else:
             #   print(f'{clustered} is an invalid input please enter Y/N')
        
#will add functions to deal with missing values and outliers
#i think i should create a second class that will deal with more advanced polynomial
#relationships and collinearity
    
        
#i need to add an option that will let the user select the best transformation for a given feature 
#and apply the transformation
        
    
#going to change the break condition from number of iterations to a set silhouette score
    
