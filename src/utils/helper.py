# ============================ Third Party libs ============================
from typing import List
import pandas
# ============================ My packages ============================
from .InputExample import InputExample


def prepare_example(data_frame: pandas.DataFrame) -> List[InputExample]:
    data = []
    for index, row in data_frame.iterrows():
        data.append(InputExample(text=row["article"],
                                 summary=row["highlights"]))
    return data
