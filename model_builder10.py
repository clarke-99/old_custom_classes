#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, PowerTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import (train_test_split, cross_val_score, KFold, 
                                     GridSearchCV, StratifiedKFold)
#import scipy.stats
#from scipy import stats
#from scipy.stats import shapiro, skew, zscore
from sklearn.metrics import (mean_squared_error, accuracy_score, 
                             f1_score, recall_score, precision_recall_curve, 
                             r2_score, roc_curve, roc_auc_score, 
                             precision_score, confusion_matrix
                            )
from scipy.stats import boxcox
from scipy.stats.mstats import normaltest
#from sklearn.pipeline import Pipeline
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import warnings
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
import copy
from tabulate import tabulate
import os
from pathlib import Path
from pandas.plotting import table
import matplotlib.gridspec as gridspec



#works with all classifiers and stacking 

class BuildModel:
    def __init__(self, data, target, model, param_grid, n_splits= None, scoring = None):
        #for building the model
        self.data = data.copy()
        self.target = target 
        self.model = copy.deepcopy(model)
        self.scale = False
        self.random_state = 42
        self.weight = False
        self.final_res = {'initial_model': None, 'feature_optimised': None,
                      'final_optimised': None}
        self.param_grid = param_grid
        self.scoring = scoring
        self.check_model = None
        self.perm_fi_select = None
        self.feat_select = None
        self.n_splits = n_splits

        
        #X and y
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.init_X_train, self.init_X_test = None, None
        self.train_index, self.test_index = None, None
        self.original_labels = pd.DataFrame()
        self.le = LabelEncoder()

     
        #Initial model
        self.initial_model = None

        #Result
        self.result_dict = {}
    
        #for feature selection
        #variables for RFE and Permutation FI
        self.importance_getter = None
        self.retained_features = None
        self.removed_features = None
        self.phase = 'initial'
        self.feat_model = None
        self.opt_model = None
            
        #results
        self.initial_result = {}
        self.feat_result = {}
        self.opt_result = {}
        self.res = {}
        self.best_params = {}
        
        
        
        
    def build(self, weight = False, scale = False, project_name = None, data_name = None):
        print("\033[1m" + '\nModel Builder\n' + "\033[0m")
        self.project_name = project_name
        self.data_name = data_name 
        self.weight = weight
        self.scale = scale
        prep_model = self.prepare_model()
        build_model = self.build_model()
        #model_eval = self.evaluate_model()
        while True:
            self.feat_select = input('Would you like to perform Recursive Feature Elimination and permutation feature importance? Y/N ')
            if self.feat_select.lower() == 'n':

                  while True:
                    self.perm_fi_select = input('Would you like to perform permutation feature importance? Y/N ')
                    
                    if self.perm_fi_select.lower() == 'y':
                        print('\n')
                        self.control()
                        
                        while True:
                            tune_params = input('Would you like to tune hyperparameters? Y/N ')
                        
                            if tune_params.lower() == 'y':
                                self.phase = 'hyperparameter_tuning'
                                self.control()
                                self.res = self.opt_result
                                opt_res = self.opt_result['error_metrics']
                                print(f'\nHyperparameter optimised results: {opt_res}')
                                return self.opt_result
                            
                            elif tune_params.lower() == 'n':
                                self.res = self.feat_result
                                res = self.feat_result['error_metrics']
        
                                print(f'\nFeature optimised results: {res}')
                                return self.feat_result
                            else:
                                return 'Invalid input. Please enter Y/N'
     
                        
                    elif self.perm_fi_select.lower() == 'n':
                        self.res = self.initial_result
                        res = self.initial_result['error_metrics']
                        print(f'\nInitial Model results: {res}')
                        return self.initial_result
            
            elif self.feat_select.lower() == 'y':
                print("\033[1m" + '\nFeature Selection\n' + "\033[0m")
                self.control()
                    
                while True:
                    tune_params = input('Would you like to tune hyperparameters? Y/N ')
                        
                    if tune_params.lower() == 'y':
                        self.phase = 'hyperparameter_tuning'
                        self.control()
                        self.res = self.opt_result
                        opt_res = self.opt_result['error_metrics']
                        print(f'\nHyperparameter optimsed results: {opt_res}')
                        return self.opt_result
                    elif tune_params.lower() == 'n':
                        self.res = self.feat_result
                        res = self.feat_result['error_metrics']
        
                        print(f'\nFeature optimsed results: {res}')
                        return self.feat_result
                    else:
                        return 'Invalid input. Please enter Y/N'
            
        res = self.res
        return res

        
    
    def prepare_model(self):
        print(f'Preparing initial {self.model} model')
        for feature in self.data.columns:
            if self.data[feature].dtype == 'object':
                print(f'Encoding feature: {feature}')
                original_labels = self.data[feature].copy()
                self.data[feature] = self.le.fit_transform(self.data[feature])
                self.original_labels = pd.DataFrame({feature: original_labels})

        
        self.X = self.data.drop(self.target, axis = 1)
        self.y = self.data[self.target]
        
        if self.scale:
            scaler = StandardScaler()
            scaled_X = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(scaled_X, columns=self.X.columns)
        self.X.reset_index(drop=True, inplace=True)
        
        
        if not self.n_splits:
            n_str = input('How many segments of training data would you like to use?')
            n = int(n_str)
            self.n_splits = n
            stratified_kf = StratifiedKFold(n_splits=n, shuffle=True, 
                                            random_state=self.random_state)

        for train_index, test_index in stratified_kf.split(self.X, self.y):
            self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.init_X_train, self.init_X_test = self.X_train, self.X_test
            self.train_index, self.test_index = train_index, test_index


        
    def build_model(self):
        #builds model
        print(f'\nBuilding initial {self.model} model')
        initial_model = self.model.fit(self.X_train, self.y_train)
        self.initial_model = initial_model
        self.check_model = initial_model
        if not self.check_model:
            return 'Model not saved'
        else:
            print(f'Model {self.check_model} saved\n')
            #predicts and scores
            self.y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            f1 = f1_score(self.y_test, self.y_pred)
            auc = roc_auc_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred)
            recall = recall_score(self.y_test, self.y_pred)
        
            #saves scores and model
            initial_result_dict = {'model': initial_model,
                           'accuracy': accuracy,
                           'f1_score': f1, 
                           'AUC_ROC': auc,
                           'precision': precision,
                           'recall': recall}
            initial_model_data = {'model': initial_model, 
                            'X_train': self.X_train,
                            'X_test': self.X_test,
                            'y_train': self.y_train,
                            'y_test': self.y_test}
        
            #creates nested dictionary for passing data further
            self.initial_result = {'model_data': initial_model_data,
                          'error_metrics':initial_result_dict}
            
            model_eval = self.evaluate_model()
            return model_eval
            
            
 
                
    def control(self): 
        if self.phase == 'initial':
            print(f'Initial Model: {self.initial_model}')
            
            if not self.scoring:
                self.scoring = input('What scoring method should be used? ')
                
            self.phase = 'feature_selection'
            
        if self.phase == 'feature_selection':
            if self.feat_select.lower() == 'y':
                print(f'Current Model: {self.check_model} ready for {self.phase}')
            elif self.feat_select.lower() == 'n' and self.perm_fi_select.lower() == 'y':
                print(f'Current Model: {self.check_model}')
            else:
                return 'Logic error from feat_select/perm_fi_select'
            return self.model_type()

        elif self.phase == 'hyperparameter_tuning':
            print("\033[1m" + '\nTuning Hyperparameters\n' + "\033[0m")
            print(f'Current Model: {self.check_model} ready for {self.phase}')
            return self.hyperparam_tuning()
        
        elif self.phase == 'completed':
            print(f'\nCurrent Phase: {self.phase}')
            print(f'Current Model: {self.check_model}')
            opt_fi = self.permutation_fi()
            return self.evaluate_model()
    
    
    
    def model_type(self):
        if self.feat_select.lower()=='y':
            try:
                if isinstance(self.initial_model, LinearSVC) or (isinstance(self.initial_model, SVC) and self.initial_model.kernel == 'linear'):
                    self.importance_getter = 'coef_'
                    return self.feature_select() 
            
                else:
                    self.importance_getter = 'auto'
                    return self.feature_select() 
                
            except ValueError: 
                print(f'{self.initial_model} incompatible with RFE - must define custom importance getter\n')
                self.retained_features = self.X.columns.tolist()
                self.permutation_fi()
                return self.rebuild_model() 
                
        else: 
            self.retained_features = self.X.columns.tolist()
            self.permutation_fi()
            return self.rebuild_model() 
            
        
        
    def feature_select(self):
        if self.phase == 'feature_selection':
            cv = StratifiedKFold(n_splits=10)
            rfecv = RFECV(estimator=self.check_model, step=1, cv=cv, scoring=self.scoring,
                      importance_getter=self.importance_getter, n_jobs = 4)
            rfecv = rfecv.fit(self.X_train, self.y_train)
        
            #updating variables and plotting figure
            self.removed_features = [feature for idx, feature in enumerate(self.X_train.columns) if not rfecv.support_[idx]]
            self.retained_features = None
            self.retained_features = rfecv.support_
            
            print(f"\nOptimal number of features based on RFECV: {rfecv.n_features_}")
            print("Removed features:", self.removed_features)
            
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel(f"Mean test {self.scoring}")
            n_scores = len(rfecv.cv_results_["mean_test_score"])
            plt.errorbar(range(1, n_scores + 1),
                         rfecv.cv_results_["mean_test_score"],
                         yerr=rfecv.cv_results_["std_test_score"])
            plt.show()
        
            #updating training data
            self.X_train = self.init_X_train.iloc[:, rfecv.support_]
            self.X_test= self.init_X_test.iloc[:, rfecv.support_]
            initial_fi = self.permutation_fi()
            feature_opt_model = self.rebuild_model()    
            return feature_opt_model
        
    def rebuild_model(self):
        if self.phase == 'feature_selection':
            print("\033[1m" + '\nBuilding Model\n' + "\033[0m")
    
            print(f'Number of features in X_train: {self.X_train.shape[1]}')
            print(f'Number of features in X_test: {self.X_test.shape[1]}\n')
            self.feat_model = self.initial_model

            self.feat_model = self.feat_model.fit(self.X_train, self.y_train)
            self.y_pred = self.feat_model.predict(self.X_test)
            self.check_model = self.feat_model     
        
            #save check_model and logic to prevent infinite loop
            self.check_model = self.feat_model
            print(f'Current Model ID: {id(self.feat_model)}')
            print(f'Current Model Check ID: {id(self.check_model)}')
            
            if self.check_model is self.feat_model:
          
                print(f'\nModel correctly updated {self.check_model} after feature selection\n')
                evaluation_results = self.evaluate_model()
                return evaluation_results
            else:
                print('Model failed to update')
                

    def evaluate_model(self):
        print("\033[1m" + f'Evaluating model: {self.check_model}' +"\033[0m")
        #print('Evaluation on full dataset')
        home = Path.home()
 
        if not self.project_name: 
            self.project_name = input('What is this project called?')
            project_name = self.project_name
        else:
            project_name = self.project_name
            
        model_type = str(self.check_model)
  
        model_name = model_type.split('(')[0]
        #print(model_name)
        
        if not self.data_name:
            self.data_name = input('What is the dataframe called? ') 
            data_name = self.data_name
        else:
            data_name = self.data_name
        
        if 'stacking' in model_name.lower():

            estimators = input('What estimators are being stacked? ')
            full_model_name = f'{model_name}({estimators})'
            
        else:
            full_model_name = self.check_model
        
        if self.X_train.shape[1] == len(self.data.drop(self.target, axis =1).columns):
            if self.phase == 'feature_selection' or self.phase == 'initial':
                file_path = f'{home}/Desktop/Coding/projects/{project_name}/{model_name}/{data_name}/n_splits:{self.n_splits}/all_features/{full_model_name}'
            elif self.phase == 'completed':
                file_path = f'{home}/Desktop/Coding/projects/{project_name}/{model_name}/{data_name}/n_splits:{self.n_splits}/all_features/optimised_{full_model_name}'
            else:
                return 'invalid phase'
        else:
            if self.phase == 'feature_selection':
                file_path = f'{home}/Desktop/Coding/projects/{project_name}/{model_name}/{data_name}/n_splits:{self.n_splits}/feat_selected/{self.X_train.shape[1]}_feats_{full_model_name}'
            elif self.phase == 'completed':
                file_path = f'{home}/Desktop/Coding/projects/{project_name}/{model_name}/{data_name}/n_splits:{self.n_splits}/feat_selected/{self.X_train.shape[1]}_feats_optimised_{full_model_name}'
            else:
                return 'invalid phase'
            
        
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        
        #train model on full dataset
        self.X = self.X[self.X_train.columns]
        y_pred = self.check_model.predict(self.X)
        y = self.y
        
        #evaluate metrics based on full dataset
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        if self.phase == 'feature_selection' and self.check_model is self.feat_model:  
            print(f'Evaluation for {model_name} after feature selection')
          
            self.feat_result_dict = {'model': self.check_model,
                           'accuracy': accuracy,
                           'f1_score': f1, 
                           'AUC_ROC': auc,
                           'precision': precision,
                           'recall': recall}


        elif self.phase == 'completed' and self.check_model is self.opt_model:
         
            print(f'Final evaluation for {model_name}')
            self.opt_result_dict = {'model': self.check_model,
                           'accuracy': accuracy,
                           'f1_score': f1, 
                           'AUC_ROC': auc,
                           'precision': precision,
                           'recall': recall}
           
            
        elif self.phase == 'initial':
            print(f'Initial evaluation for {model_name}')
            #return 'Continuing'
        else: 
            return 'Invalid phase'
            
        #decoding target feature if needed
        if not self.original_labels.empty:
            if self.target in self.original_labels.columns.tolist():
                
                #decoding is working
                print(f'\nDecoding {self.target}')
                class_labels = self.le.classes_

                y_pred = self.le.inverse_transform(y_pred)
                y = self.le.inverse_transform(self.y)
    
        else:
            y = self.y
            y_pred = y_pred
            
        if 'stacking' in model_name.lower():
            model_name = full_model_name
        else:
            model_name = model_name
            
            
        metrics_df = pd.DataFrame({'Model': [model_name],
                                   'Stratified splits': [self.n_splits],
                                   'Features' : [self.X_train.shape[1]],
                                   'Accuracy': [accuracy],
                                   'Precision': [precision],
                                   'Recall': [recall],
                                   'F1 Score': [f1],
                                   'AUC-ROC': [auc]},)
        metrics_df.set_index('Model', inplace=True)
        

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(25, 20))

        # Create a gridspec to allow customizing the layout
        gs = gridspec.GridSpec(nrows = 2,ncols = 1, height_ratios=[9, 1])

        # Confusion matrix heatmap
        ax0 = plt.subplot(gs[0])
    
        heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax0,
                              annot_kws={'size': 40}, square = True, cbar = False)

        
        ax0.set_xlabel("Predicted", fontsize=40)
        ax0.set_ylabel("Actual", fontsize=40)
        ax0.set_xticks(list(i + 0.5 for i in range(len(class_labels))))
        ax0.set_yticks(list(i + 0.5 for i in range(len(class_labels))))
        
        
        ax0.set_xticklabels(class_labels, rotation=0, ha='center', fontsize=40)
        ax0.set_yticklabels(class_labels, rotation=90, ha='left', fontsize=40,
                           position=(-0.04, 0.5))
        
        
        ax0.set_title(f"""Confusion Matrix and Error Metrics for {full_model_name} based on {data_name} dataset""",
                      fontsize=45)
        
        
        # Error metrics table
        ax1 = plt.subplot(gs[1])
        ax1.axis('off')
        
        rounded_metrics_df = metrics_df.round(5)
        tab = table(ax1, rounded_metrics_df, loc='center', colWidths=[0.3]*len(rounded_metrics_df.columns))
        tab.auto_set_font_size(False)
        tab.set_fontsize(40)
        tab.scale(1, 5)
        tab.auto_set_column_width(col=list(range(len(metrics_df.columns))))

        plt.tight_layout()

        combined_file_path = file_path + '_combined.png'
        plt.savefig(combined_file_path, bbox_inches='tight', pad_inches=0.1)
        plt.show()


        save_model = self.save_models()
        return save_model
    
    
    def save_models(self):
        if self.check_model is self.feat_model and self.phase == 'feature_selection':
            print(f'Saving feature optimised {self.check_model} model')
            print(f'Number of features: {self.X_train.shape[1]}\n')

            
            #might not need X and y values
            feat_model_res = {'model': self.feat_model, 
                            'X_train': self.X_train,
                            'X_test': self.X_test,
                            'y_train': self.y_train,
                            'y_test': self.y_test}
            feat_err_dict = self.feat_result_dict

            self.feat_result = {'model_data': feat_model_res,
                               'error_metrics': feat_err_dict}

            feat_res = self.feat_result['error_metrics']
            print(f'{self.phase} results: {feat_res}\n')
            
            return self.feat_result

            
        elif self.phase == 'completed':
            print(f'\nSaving hyperparameter optimised {self.check_model} model')
            print(f'Parameters {self.best_params}\n')
            self.phase = 'feature_selection'
            opt_model_res = {'model': self.opt_model, 
                            'X_train': self.X_train,
                            'X_test': self.X_test,
                            'y_train': self.y_train,
                            'y_test': self.y_test}
            opt_err_dict = self.opt_result_dict
 
            self.opt_result = {'model_data': opt_model_res,
                               'error_metrics': opt_err_dict,}
            opt_result = self.opt_result['error_metrics']
       
            if not opt_result:
                print('Result not saved')
            else:
                return self.opt_result
        
            
    def hyperparam_tuning(self):
        
        search = GridSearchCV(self.check_model, self.param_grid, cv = 5,
                              scoring=self.scoring, n_jobs = 4)
        search.fit(self.X_train, self.y_train)
        self.best_params = search.best_params_
        best_score = search.best_score_
        
        #rebuild model
        self.opt_model = self.check_model.set_params(**self.best_params)
        self.opt_model.fit(self.X_train, self.y_train)
        self.y_pred = self.opt_model.predict(self.X_test)
                
        #check model
        self.check_model = self.opt_model
        
        if self.check_model is self.opt_model:
            print(f'Model correctly updated to {self.check_model} after tuning')
            self.phase = 'completed'
            print(f'\nBest parameters: {self.best_params}')
            print(f'Best score: {best_score}')
            return self.control()
        else:
            message = 'Model not updated'
            return message


    def permutation_fi(self):
        
        print("\033[1m" +'\nCalculating Permutation Feature Importance' +"\033[0m")
        
        if self.phase == 'feature_selection': 
            X_test = self.X_test
 
            self.initial_model = self.initial_model.fit(self.X_train, self.y_train)
            model = self.initial_model
            print(f'\nModel: {self.check_model}')

        elif self.phase == 'completed':
            if self.param_grid:
                    X_test = self.X_test
                    print(f'\nModel: {self.check_model}')
                    model = self.opt_model


            self.opt_model = self.opt_model.fit(self.X_train, self.y_train)
            model = self.opt_model
       
    
        print(f'Number of features: {X_test.shape[1]}\n')
        print(f'Permutation feature importance for {model}\n')     
            
        r = permutation_importance(model, X_test, self.y_test,
                                   n_repeats=30, random_state=self.random_state,
                                  n_jobs = 4)
        for i in r.importances_mean.argsort()[::-1]:
            print(f"{X_test.columns[i]:<8}"
                  f" {r.importances_mean[i]:.5f}"
                  f" +/- {r.importances_std[i]:.3f}")
            
        
        while True:
            plot_fig = input('\nWould you like to visualise permutation feature importance? Y/N ')
            if plot_fig.lower() == 'y':        
                plt.figure(figsize=(10, 10))
                plt.boxplot(r.importances.T, vert=False, labels=X_test.columns)
                plt.title(f"Permutation Feature Importance for {model}")
                plt.xlabel("Importance Score")
                plt.show()
                break
            elif plot_fig.lower() == 'n':
                print('\nContinuing\n')
                break
            else: 
                print('Invalid input. Please enter "Y" for yes or "N" for no.')

        feature_importance_scores = r.importances_mean

        
        if self.phase == 'feature_selection':
            while True:
                question = input('Would you like to remove more features based on their feature importance? Y/N ')
                if question.lower() == 'y':
                    self.manual_select(feature_importance_scores) 
                    break  
                elif question.lower() == 'n':
                    return '\nContinuing'
                else:
                    print('Invalid input. Please enter "Y" for yes or "N" for no.')
        
    #allows removal of features based on their importance score
    def manual_select(self, feature_importance_scores):
        # Sort features by importance scores
        sorted_indices = feature_importance_scores.argsort()

        
        while True:
            threshold_input = input('Below what importance should features be removed: ')
            
            try:
                threshold = float(threshold_input)
                if threshold >= 0:
                    #when using an SVC model that has not gone through feature select
                    if isinstance(self.initial_model, SVC) and self.initial_model.kernel != 'linear' or self.feat_select.lower() == 'n':
                        # Create a boolean mask for the selected features
                        selected_features = (feature_importance_scores >= threshold)
                        # Use the boolean mask to select the retained features
                        self.X_train = self.init_X_train.loc[:, selected_features]
                        self.X_test = self.init_X_test.loc[:, selected_features]
                        # Update retained features based on the mask
                        self.retained_features = self.X_train.columns.tolist()
                    else:
                        features_to_remove = sorted_indices[abs(feature_importance_scores) < threshold]
                        self.retained_features[features_to_remove] = False
                        self.X_train = self.init_X_train.iloc[:, self.retained_features]
                        self.X_test = self.init_X_test.iloc[:, self.retained_features]
                    break
                else:
                    print('Threshold must be positive')
            except ValueError:
                print('Invalid input. Please enter a valid numeric threshold.')
                
            return self.evaluate_model()
                

