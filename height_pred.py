import joblib
import numpy as np

model = joblib.load("./height_classifier.pkl")


class Height():
    def __init__(self, height=160, weight=80):
        self.height = height - 100
        self.weight = weight

    def predict(self):
        return model.predict(np.array([self.height, self.weight]).reshape(1, -1))[0]