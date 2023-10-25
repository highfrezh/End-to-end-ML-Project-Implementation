from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_preprocessing import DataPreprocessing
from mlProject import logger
from pathlib import Path


STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_preprocessing_config = config.get_data_preprocessing_config()
                data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
                data = data_preprocessing.remove_duplicate()
                result = data_preprocessing.is_null_present(data)
                if result == True:
                    data = data_preprocessing.impute_missing_values(data)
                col_to_drop = data_preprocessing.get_columns_with_zero_std_deviation(data)
                data = data_preprocessing.remove_columns(data=data, columns=col_to_drop)
                data_preprocessing.save_preprocessed_data(data=data)
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)
