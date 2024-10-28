from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp  # Used to check for data_drift.
from sensor.utils.main_utils import read_yaml_file, write_yaml_file
import pandas as pd
import os,sys

class DataValidation:


    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                       data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact= data_ingestion_artifact
            self.data_validation_config= data_validation_config
            self._schema_config=  read_yaml_file(SCHEMA_FILE_PATH)  # protected variable (_ is used) 
        except Exception as e:
            raise SensorException(e, sys)
    
    """
    The @staticmethod decorator in Python defines a method that doesn't rely on the instance (object) of a class. 
    This means it can be called on the class itself, without needing an instance of that class.
    """
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:

        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)
    
    def detect_dataset_drift(self, base_df, current_df, threshold= 0.5) -> bool:
        try: 
            status= True
            report= {}
            for column in base_df.columns:
                b_df= base_df[column]
                c_df= current_df[column]
                is_same_dist= ks_2samp(b_df, c_df)
                if threshold<=is_same_dist.pvalue :
                    is_found= False
                else:   
                    is_found= True
                    status= False
                report.update({
                column: {
                'p_value': float(is_same_dist.pvalue),
                'drift_status': is_found}
                    })
            drift_report_file_path= self.data_validation_config.drift_report_file_path 

            # Creating Directory
            dir_path= os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok= True)
            write_yaml_file(file_path= drift_report_file_path, content= report, )
            return status
        except Exception as e:
            raise SensorException(e, sys)
    
    def validate_number_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns= len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise SensorException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame)-> bool:
        try:
            numerical_columns= self._schema_config['numerical_columns']
            dataframe_columns= dataframe.columns
             
            numerical_columns_present= True
            missing_numerical_columns= []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(num_column)
            
            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
            return numerical_columns_present
            
        except Exception as e:
            raise SensorException(e, sys)
    

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message= ""
            train_file_path= self.data_ingestion_artifact.trained_file_path
            test_file_path= self.data_ingestion_artifact.test_file_path

            #Reading data from train and test file location
            train_dataframe= DataValidation.read_data(train_file_path)
            test_dataframe= DataValidation.read_data(test_file_path)

            #Validate number of columns
            
            status= self.validate_number_columns(dataframe= train_dataframe)
            if not status:
                error_message= f"{error_message}Train dataframe does't contain all columns.\n"

            status= self.validate_number_columns(dataframe= test_dataframe)
            if not status:
                error_message= f"{error_message}Test dataframe does't contain all columns.\n"
                
            #Validate numerical columns

            status = self.is_numerical_column_exist(dataframe= train_dataframe)
            if not status:
                error_message= f"{error_message}Train dataframe does't contain all numerical columns.\n"
            
            status = self.is_numerical_column_exist(dataframe= test_dataframe)
            if not status:
                error_message= f"{error_message}Test dataframe does't contain all numerical columns.\n"
            
            if len(error_message)>0:
                raise Exception(error_message)
            
            #Let check data drift
            

            status= self.detect_dataset_drift(base_df= train_dataframe, current_df= test_dataframe)
            
            # Create data validation articact
            if status == False:
                dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
                
                os.makedirs(dir_path, exist_ok=True)
                
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, index=False, header=True
                )

                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path, index=False, header=True
                )
            else:
                dir_path = os.path.dirname(self.data_validation_config.invalid_train_file_path)
                
                os.makedirs(dir_path, exist_ok=True)
                
                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path, index=False, header=True
                )

                test_dataframe.to_csv(
                    self.data_validation_config.invalid_test_file_path, index=False, header=True
                )
                

            data_validation_artifact= DataValidationArtifact(
                validation_status= status,
                valid_train_file_path= self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path= self.data_ingestion_artifact.test_file_path,
                invalid_test_file_path= None,
                invalid_train_file_path= None,
                drift_report_file_path= self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)
        

        '''
            `Training Pipeline`
            - Train and Test dataset must have same distribution.
            - Base dataset = train.csv  is ---->
            -  ----> To compare with= test.csv
            - If they have smae distribution: No Datadrift present
            - `Solution`: If there is Datadrift [distriution not same] present do train test split again.

            `Instance Prediction`
            - Not possible to detect datadrift instance immediately because it is not possible to summarize one record.
            - `Solution`
            - You can save each request in database. Then you can fetch all request by hour or by day.
            - `Example`: collect data of one day.
            - go for Data drift detection
            - train dataset, collected data set see the datdrift report if huge difference go for retraining.
            - else check for data drift every day.
            
        '''
    