import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np

from sklearn.linear_model import LogisticRegression
from deep_autoencoder import DeepAutoencoder

class ScoreFeatureRelevance:
    def __init__(self, data_normal, data_anomalies, epochs=10, batch_size=5, learning_rate=0.05):
        self.data_normal    = data_normal
        self.data_anomalies = data_anomalies

        self.y_normal    = np.ones(len(self.data_normal))
        self.y_anomalies = np.zeros(len(self.data_anomalies))

        self.X  = np.concatenate((self.data_normal, self.data_anomalies))
        self.Y  = np.concatenate((self.y_normal, self.y_anomalies), axis=None)

        self.scores         = []

        self.dim = len(data_anomalies[0])

        self.epochs         = epochs
        self.batch_size     = batch_size
        self.learning_rate  = learning_rate

        self.autoencoder_config = {
            "o_activation": "sigmoid",
            "optimizer": {
                "name": "adam",
                "learning_rate": self.learning_rate
            },
            "epochs": self.epochs,
            "batch_size": batch_size,
            "input_size": self.dim,
            "bias": 1.0,
            "loss": "mse",
            "encoding_layers": [
                {
                    "size": self.dim,
                    "activation": "relu"
                },
                {
                    "size": int(round((2/3) * self.dim)),
                    "activation": "relu"
                }
            ],
            "decoding_layers": [
                {
                    "size": int(round((2/3) * self.dim)),
                    "activation": "relu"
                },
                {
                    "size": self.dim,
                    "activation": "sigmoid"
                }
            ]
        }

        self.autoencoder = DeepAutoencoder(self.autoencoder_config)

    def run(self):
        self.autoencoder.compile()
        self.autoencoder.summary()
        self.autoencoder.train(self.X)

        self.Z  = self.autoencoder.encode(self.X)

        self.Z_anomalies    = self.autoencoder.encode(self.data_anomalies)

        clf             = LogisticRegression(random_state=0).fit(self.Z, self.Y)
        probabilities   = clf.predict_proba(self.Z_anomalies)

        clf_pd              = LogisticRegression(random_state=0).fit(self.X, self.Y)
        pd_probabilities    = clf_pd.predict_proba(self.data_anomalies)

        for i in range(len(probabilities)):
            temp_score = {
                "anomaly_vector": self.data_anomalies[i],
                "fr_classification": "",
                "fr_score": 0.00
            }

            p       = probabilities[i][1]
            p_pd    = pd_probabilities[i][1]

            fr_score = p / p_pd

            if fr_score >= 0 and fr_score < 0.25:
                temp_score["fr_classification"] = "VERY_HARD"
            elif fr_score >= 0.25 and fr_score < 0.5:
                temp_score["fr_classification"] = "HARD"
            elif fr_score >= 0.5 and fr_score < 0.75:
                temp_score["fr_classification"] = "MEDIUM"
            elif fr_score >= 0.75:
                temp_score["fr_classification"] = "EASY"

            temp_score["fr_score"] = fr_score

            self.scores.append(temp_score)
