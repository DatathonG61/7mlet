import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import BayesianRidge

class ContextualThompsonSampling:
    def __init__(self):
        self.offers = [
            "CARTAO_CREDITO",
            "INVESTIMENTO"
        ]

        self.models = {
            offer: self._create_model()
            for offer in self.offers
        }



    def _create_model(self):
        categorical_features = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome"
        ]
        numerical_features = [
            "age",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "foi_contatado_antes"
        ]

        transformer = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                ),
                (
                    "numerical",
                    "passthrough",
                    numerical_features
                )
            ]
        )

        return Pipeline(
            [
                ("features", transformer),
                ("model", BayesianRidge())
            ]
        )

    def train(self, x, offer, reward):
        self.models[offer].fit(x, reward)

    def recommend(self, customer):
        samples = {}

        for offer, model in self.models.items():
            prediction, uncertainty = model.predict(customer, return_std=True)

            sample = np.random.normal(prediction, uncertainty)

            samples[offer] = sample

        return max(samples, key=samples.get)