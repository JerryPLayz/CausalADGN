import pandas as pd
from pathlib import Path
from typing import Union

def save_history(
        history: dict[str, list[float]],
        path: Union[Path, str]
):
    """
    Saves the training history to a csv file.
    An epoch column is prepended for clarity.
    :param history: dict[metric_name, list[per_epoch_value]]
    :param path:  File path to write the CSV to. Parent directories created if absent.
    :return: The dataframe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(len(df)))

    df.to_csv(path, index=False)
    return df
