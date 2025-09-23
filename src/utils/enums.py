from enum import Enum

from src.model.movie_predictor import MoviePredictor


class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]

    @classmethod
    def validation(cls, name: str):
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")


class ModelTypes(CustomEnum):
    MOVIE_PREDICTOR = MoviePredictor