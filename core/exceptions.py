class AppException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ConfigError(AppException):
    pass


class ClassifierException(AppException):
    pass


class ESResponseEmpty(AppException):
    pass


class ScoreTooLow(ClassifierException):
    pass


class AnswerNotFound(ClassifierException):
    pass


# class ClassifierTimeout(ClassifierException):
#     pass
