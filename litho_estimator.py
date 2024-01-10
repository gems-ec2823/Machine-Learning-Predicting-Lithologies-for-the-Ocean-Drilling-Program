import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# If you created custom transformers or helper functions, you can also add them to this file.

class LithoEstimator:
    '''Used to predict lithology in IODP wells. The signature (method name, argument and return types) for the strict minimum number of methods needed are already written for you below.
    Simply complete the methods following your own notebook results. You can also add more methods than provided below in order to keep your code clean.'''

    def __init__(self, path:str='data/log_data.csv') -> None:
        '''The path is a path to the training file. The default is the file I gave you.
        You want to create an X_train, X_test, y_train and y_test following the same principle as in your
        notebook. You also want to define and train your estimator as soon as you create it.
        
        I recommend creatubg the following instance variables in your __init__ method:
        self.X_train, self.X_test, self.y_train, self.y_test
        self.encoder - the label encoder for your categories
        self.model - the entire trained model pipeline

        Note that this class should not handle hyperparameter searching or feature selection - if you did those in your Part B 
        simply use your best estimators.
        
        '''
        df = pd.read_csv(path)
        df = df.drop_duplicates()
        sorted_data = df.sort_values(by='DEPTH_WMSF')

        split_index = int(sorted_data.shape[0] * 0.7)
        df_train = sorted_data.iloc[:split_index]
        df_test = sorted_data.iloc[split_index:]
        self.X_train = df_train.drop(columns = 'lithology')
        self.X_test = df_test.drop(columns = 'lithology')
        self.y_train = df_train['lithology']
        self.y_test = df_test['lithology']

     
            
        prepro_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),  
            MinMaxScaler()  
                        )
            
        preprocessor = ColumnTransformer(
            transformers=[
            ('num', prepro_pipeline, ['DEPTH_WMSF', 'HTHO', 'HURA', 'IMPH', 'SFLU']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['munsel_color'])
            ])


        
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000))
        ])

        self.model.fit(self.X_train, self.y_train)

    def x_test_score(self):
        '''Returns the F1 macro score of the X_test. This should be of type float.'''
        y_pred = self.model.predict(self.X_test)
        
        return f1_score(self.y_test, y_pred, average='macro')

        

    def get_Xs(self) -> (pd.DataFrame, pd.DataFrame):
        '''Returns the X_train and X_test. This method is already written for you.'''

        return self.X_train, self.X_test
    
    def get_ys(self) -> (pd.DataFrame, pd.DataFrame):
        '''Returns the y_train and y_test. This method is already written for you.'''

        return self.y_train, self.y_test

    def predict(self, path_to_new_file:str='data/new_data.csv') -> np.array:
        '''Uses the trained algorithm to predict and return the predicted labels on an unseen file.
        The default file is the unknown_data.csv file in your data folder.
        
        Return a numpy array (the default for the "predict()" function of sklearn estimator)'''

        new_data = pd.read_csv(path_to_new_file)
        return self.model.predict(new_data)

    def get_model(self) -> Pipeline:
        '''returns the entire trained pipeline, i.e. your model. 
        This will include the data preprocessor and the final estimator.'''

        return self.model