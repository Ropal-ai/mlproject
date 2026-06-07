import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # ensure features is a DataFrame
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            # normalize missing markers (convert None to np.nan)
            features = features.replace({None: np.nan})

            # find categorical transformer and its columns from the fitted ColumnTransformer
            cat_pipeline = None
            cat_cols = []
            for name, transformer, cols in getattr(preprocessor, "transformers_", []):
                if name == "cat_pipeline":
                    cat_pipeline = transformer
                    cat_cols = cols
                    break

            # if categorical pipeline exists, fill missing and unknown categories
            if cat_pipeline is not None and len(cat_cols) > 0:
                # get fitted imputer and one-hot encoder if present
                imputer = None
                ohe = None
                if hasattr(cat_pipeline, "named_steps"):
                    imputer = cat_pipeline.named_steps.get("imputer")
                    ohe = cat_pipeline.named_steps.get("one_hot_encoder")

                stats = None
                if imputer is not None and hasattr(imputer, "statistics_"):
                    stats = list(imputer.statistics_)

                if ohe is not None and hasattr(ohe, "categories_"):
                    for i, col in enumerate(cat_cols):
                        try:
                            allowed = list(ohe.categories_[i])
                        except Exception:
                            allowed = []
                        fill = None
                        if stats is not None and i < len(stats):
                            fill = stats[i]
                        elif len(allowed) > 0:
                            fill = allowed[0]

                        # fill missing values
                        if fill is not None:
                            features[col] = features[col].fillna(fill)

                        # replace unknown categories with fill
                        if len(allowed) > 0 and fill is not None:
                            features[col] = features[col].where(features[col].isin(allowed), other=fill)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)