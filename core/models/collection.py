from pydantic import BaseModel


class MLModel(BaseModel):
    name: str
    local_path: str
    model: object


class ModelsCollection:
    def __init__(self):
        self.models: dict[str, MLModel] = {}

    def add_model(self, model: MLModel) -> None:
        if model.name in self.models:
            raise ValueError(f"Model {model.name} already exists.")
        self.models[model.name] = model

    def get_model(self, model_name: str) -> MLModel | None:
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found.")
        return model

    def remove_model(self, model_name: str) -> None:
        if model_name in self.models:
            del self.models[model_name]

    def update_model(self, model_name: str, **kwargs):
        model_info = self.models.get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found.")
        for key, value in kwargs.items():
            setattr(model_info, key, value)

    def download_model(self, model_name: str):
        pass
