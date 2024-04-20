from enum import Enum

class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class AvailableEmbedders(str, ExtendedEnum):
    LSTM = "lstm"
    BI_LSTM = "bi-lstm"
    BI_LSTM_POOL = "bi-lstm-pool"
    MEAN = "mean"
