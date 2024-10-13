import joblib


class Model:
    def init(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, X):
        return self.model.predict(X)[0]
