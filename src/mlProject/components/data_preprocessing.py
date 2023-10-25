import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer
from mlProject import logger
from mlProject.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    """
        This class shall  be used to clean and transform the data before training.
    """

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def remove_duplicate(self):
        """
            Method Name: remove_duplicate
            Description: This method checks whether there are duplicate record in the pandas Dataframe or not.
            Output: Returns a Boolean Value. True if Duplicate record are present in the DataFrame, False if they are not present.
            On Failure: Raise Exception
        """
        logger.info('Entered the remove_duplicate method of the Preprocessor class')
        data =  pd.read_csv(self.config.data_path)
        try:
            if data.duplicated().sum() > 0:
                # Using DataFrame.drop_duplicates() to keep first duplicate row
                new_data = data.drop_duplicates(keep='first')
                logger.info(f"{data.duplicated().sum()} found and {data.duplicated().sum() - 1} are removed")
                return new_data
            else:
                logger.info("No duplicate record found!!!")
                return data                       
        except Exception as e:
            logger.info('Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()
        
        
    def is_null_present(self, data):
        """
            Method Name: is_null_present
            Description: This method checks whether there are null values present in the pandas Dataframe or not.
            Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
            On Failure: Raise Exception
        """
        logger.info('Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.data = data
        try:
            self.null_counts=self.data.isna().sum() # check for the count of null values per column
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            if(self.null_present):
                logger.info('Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
                return self.null_present
        except Exception as e:
            logger.info('Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()
        

    def impute_missing_values(self, data):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
            Output: A Dataframe which has all the missing values imputed.
            On Failure: Raise Exception
        """
        logger.info('Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data=pd.DataFrame(data=self.new_array, columns=self.data.columns)
            logger.info('Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            logger.info('Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()
        
    
    def get_columns_with_zero_std_deviation(self,data):
        """
            Method Name: get_columns_with_zero_std_deviation
            Description: This method finds out the columns which have a standard deviation of zero.
            Output: List of the columns with standard deviation of zero
            On Failure: Raise Exception
        """
        logger.info('Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            logger.info('Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            logger.info('Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

        

    def remove_columns(self, data, columns):
        """
            Method Name: remove_columns
            Description: This method removes the given columns from a pandas dataframe.
            Output: A pandas DataFrame after removing the specified columns.
            On Failure: Raise Exception
        """
        logger.info('Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            logger.info('Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            logger.info('Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            logger.info('Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()
        

    def save_preprocessed_data(self,data):
        self.data = data
        self.data.to_csv(os.path.join(self.config.root_dir, "preprocessed_data.csv"),index = False)
        logger.info(f"Clean data saved successfully!!!!. to {self.config.root_dir}")
        logger.info(self.data.shape)
        print(self.data.shape)
        
        

    # def separate_label_feature(self, data, label_column_name):
    #     """
    #         Method Name: separate_label_feature
    #         Description: This method separates the features and a Label Coulmns.
    #         Output: Returns two separate Dataframes, one containing features and the other containing Labels .
    #         On Failure: Raise Exception
    #     """
    #     self.data = data
    #     logger.info('Entered the separate_label_feature method of the Preprocessor class')
    #     try:
    #         self.X=self.data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
    #         self.Y=self.data[label_column_name] # Filter the Label columns
    #         logger.info('Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
    #         return self.X,self.Y
    #     except Exception as e:
    #         logger.info('Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))            
    #         logger.info('Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
    #         raise Exception()


    