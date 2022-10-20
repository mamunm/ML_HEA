"""
A script to select Machine Learning Models for Yield Strength Prediction.
"""

# Importing required packages
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Input:
    # Data: Python Dataframe with the true labels as its last column.

# Class Object to define all the methods
class Yield_Strength:
    def __init__(self,data):
        self.features = data.iloc[:,2:data.shape[1]-1]                                      # Extracting the features
        self.labels = data.iloc[:,-1]                                                       # Extracting the labels
        self.colnames = self.features.columns                                               # Defining the feature set column names
        self.alloy_names = data.iloc[:,0]                                                   # Extracting the alloy names

    def preprocessing(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()                                                      # Calling the standard scaler object
        self.scaled_features = pd.DataFrame(self.scaler.fit_transform(self.features))       # Standardizing the features
        self.scaled_features.columns = self.colnames    
        print("Scaled Features:")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        print(self.scaled_features.head())
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        
    # Now we will define the machine learning models.
    # All three of the ML models will require a separate input.
    # Input:
        # params: Python Dictionary with parameters specified.


    def optimizer_gbr(self,grid_values):
        #model
        model = GradientBoostingRegressor(random_state = 1)
        # Gridsearch
        clf = GridSearchCV(model, grid_values, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs=-1,return_train_score = True)                # Calling the Grid Search object
        clf.fit(self.scaled_features,self.labels)                                                           # Fitting the grid search object
        print('Best Parameters: ', clf.best_params_)
        print('Best Score: ',clf.best_score_)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        result = pd.DataFrame(clf.cv_results_)
        result.to_csv('CV_result for gradient boost')
        print('CV result saved as .csv')
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        self.best_params = clf.best_params_
        opt_model = GradientBoostingRegressor(**clf.best_params_,random_state = 1)
        self.fitted_model = opt_model.fit(self.scaled_features,self.labels)
        self.model_name = 'Gradient Boosting'
        return clf.best_params_, clf.best_score_
        
    
    def optimizer_ada(self,grid_values):
        #model
        model = AdaBoostRegressor(random_state = 1)

        # Gridsearch
        clf = GridSearchCV(model, grid_values, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs=-1,return_train_score = True)                # Calling the Grid Search object
        clf.fit(self.scaled_features,self.labels)                                                           # Fitting the grid search object
        print('Best Parameters: ', clf.best_params_)
        print('Best Score: ',clf.best_score_)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        result = pd.DataFrame(clf.cv_results_)
        result.to_csv('CV_result for Adaboost')
        print("CV result saved as .csv")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        self.best_params = clf.best_params_
        opt_model = AdaBoostRegressor(**clf.best_params_,random_state = 1)
        self.fitted_model = opt_model.fit(self.scaled_features,self.labels)
        self.model_name = 'Ada Boost'
        return clf.best_params_, clf.best_score_

    def optimizer_xgb(self,grid_values):
        #model
        model = xgb.XGBRegressor(random_state = 1)
         
        # Gridsearch
        clf = GridSearchCV(model, grid_values, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs=-1,return_train_score = True)                # Calling the Grid Search object
        clf.fit(self.scaled_features,self.labels)                                                           # Fitting the grid search object
        print('Best Parameters: ', clf.best_params_)
        print('Best Score: ',clf.best_score_)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        result = pd.DataFrame(clf.cv_results_)
        result.to_csv('CV_result for XGboost')
        print(" CV results saved as .csv.")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        self.best_params = clf.best_params_
        opt_model = xgb.XGBRegressor(**clf.best_params_,random_state=1)
        self.fitted_model = opt_model.fit(self.scaled_features,self.labels)
        self.model_name = 'XG Boost'
        return clf.best_params_, clf.best_score_
        
    def optimizer_RF(self,grid_values):
        #model
        model = RandomForestRegressor(random_state = 1)
        # Gridsearch
        clf = GridSearchCV(model, grid_values, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs=-1,return_train_score = True)                # Calling the Grid Search object
        clf.fit(self.scaled_features,self.labels)                                                           # Fitting the grid search object
        print('Best Parameters: ', clf.best_params_)
        print('Best Score: ',clf.best_score_)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        result = pd.DataFrame(clf.cv_results_)
        result.to_csv('CV_result for Random Forest')
        print('CV result saved as .csv')
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        self.best_params = clf.best_params_
        opt_model = RandomForestRegressor(**clf.best_params_,random_state = 1)
        self.fitted_model = opt_model.fit(self.scaled_features,self.labels)
        self.model_name = 'Random Forest'
        return clf.best_params_, clf.best_score_
    
    def optimizer_LASSO(self,grid_values):
            #model
            model = LassoCV(random_state = 1)
             
            # Gridsearch
            clf = GridSearchCV(model, grid_values, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs=-1,return_train_score = True)                # Calling the Grid Search object
            clf.fit(self.scaled_features,self.labels)                                                           # Fitting the grid search object
            print('Best Parameters: ', clf.best_params_)
            print('Best Score: ',clf.best_score_)
            print("----------------------------------------------------------------------------------------------------------------------------------------")
            result = pd.DataFrame(clf.cv_results_)
            result.to_csv('CV_result for XGboost')
            print(" CV results saved as .csv.")
            print("----------------------------------------------------------------------------------------------------------------------------------------")
            self.best_params = clf.best_params_
            opt_model = LassoCV(**clf.best_params_,random_state=1)
            self.fitted_model = opt_model.fit(self.scaled_features,self.labels)
            self.model_name = 'LASSO'
        return clf.best_params_, clf.best_score_
        
   def optimizer_SVR(self,grid_values):
        #model
        model = SVR(kernel='linear')
             
        # Gridsearch
        clf = GridSearchCV(model, grid_values, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs=-1,return_train_score = True)                # Calling the Grid Search object
        clf.fit(self.scaled_features,self.labels)                                                           # Fitting the grid search object
        #print('Best Parameters: ', clf.best_params_)
        print('Best Score: ',clf.best_score_)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        result = pd.DataFrame(clf.cv_results_)
        result.to_csv('CV_result for SVR')
        print(" CV results saved as .csv.")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        self.best_params = clf.best_params_
        opt_model = SVR(**clf.best_params_)
        self.fitted_model = opt_model.fit(self.scaled_features,self.labels)
        self.model_name = 'SVR'
        return clf.best_params_, clf.best_score_    
	   
    def feature_importance(self):
        importance = list(self.fitted_model.feature_importances_)                               # Extracting the feature importances
        imp = pd.DataFrame({'features': self.colnames,'importance':importance})                  # Creating a dataframe with feature names and importances
        imp = imp.sort_values(by = 'importance',axis = 0, ascending = False)                        # sorting the dataframe based on importances
        
        # Visualization
        plt.figure(figsize = (8,8))
        sns.barplot(imp.loc[:,'importance'],imp.loc[:,'features'],palette="Blues_d")        # Generating a barplot with feature importances
        plt.title("Feature Importance For " + self.model_name)                                   # plot title
        plt.tight_layout()                      
        plt.savefig("feature_imp_"+self.model_name+".png")                                       # saving the plot to the directory
        plt.show()

# Testing set
    # Input:
        # Validation output with same format as the training dataset.
    # Output:
        # Creates .csv file with the predicted output
        # Returns the error in prediction for each output.
    def test(self,validation_data):
        # Preprocessing the test data set
        validation_alloy_name = validation_data.iloc[:,1]                                              # Extracting the validation set alloy names
        validation_features = validation_data.iloc[:,2:validation_data.shape[1]-1]                      # Extracting the validation set features
        validation_labels = validation_data.iloc[:,-1]                                              # Extracting the validation set labels
        scaled_validation = pd.DataFrame(self.scaler.transform(validation_features))                    # using the scaler method to standardize the validation data 
        scaled_validation.columns = self.colnames
        print("Scaled Validation Set: ")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        print(scaled_validation.head())
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        
        pred = self.fitted_model.predict(scaled_validation)                 # Predicting the labels
        print('pred', pred.shape)
        error = mean_absolute_error(validation_labels,pred)                                         # calculating the MAE error
        print("MAE Error for " + self.model_name,' : ',error)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        
        pred = self.fitted_model.predict(scaled_validation)                 # Predicting the labels
        print('pred', pred.shape)
        R2 = r2_score(validation_labels,pred)
        print("r2 Error for " + self.model_name,' : ',R2)
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        pred = self.fitted_model.predict(scaled_validation)                 # Predicting the labels
        print('pred', pred.shape)
        rmse_val = mean_squared_error(validation_labels,pred,  squared=False)
        print("rmse Error for " + self.model_name,' : ',rmse_val)
        print("----------------------------------------------------------------------------------------------------------------------------------------")



        # Generating a dateframe with predictions from each ML model and  
        print("Generating the resulting CSV output.")
        pred = pd.DataFrame(pred)
        predictions = pd.concat([validation_alloy_name,validation_labels,pred],axis = 1)
        predictions.columns = ['Alloy','True Yield Strength',self.model_name]
        csv_name = 'Predictions_'+self.model_name+'.csv'
        predictions.to_csv(csv_name)
        return predictions
