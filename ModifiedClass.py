import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier

class MyClassifier(Classifier):
        def load_data(self, file, amount=-1):
            data, self.attrs = use_dataframe(file, self.attrs, self.label, self.numeric, amount)
            return data
        

def use_dataframe(file, attrs, label, numeric, amount):
    # Read CSV more robustly: use the python engine and keep all fields as strings
    # to avoid tokenization errors on malformed rows. Warn on bad lines.
    try:
        df = pd.read_csv(file, engine='python', on_bad_lines='warn', dtype=str)
    except TypeError:
        # Older pandas versions used error_bad_lines / warn_bad_lines; fall back
        df = pd.read_csv(file, engine='python', dtype=str)
    except Exception:
        # As a final fallback, try skipping bad lines
        df = pd.read_csv(file, engine='python', on_bad_lines='skip', dtype=str)

    # Ensure all requested attribute columns exist; if missing, create them with empty values
    for col in attrs:
        if col not in df.columns:
            df[col] = ''

    # Ensure label column exists
    if label not in df.columns:
        df[label] = ''

    # Select columns in the requested order and concatenate label as last column
    df_x = df.reindex(columns=attrs)
    df_y = df[label]
    result = pd.concat([df_x, df_y], axis=1)

    # Convert DataFrame to list-of-lists (rows) so downstream code that expects lists works
    rows = result.values.tolist()

    # Keep existing behavior of appending label to attrs for compatibility
    attrs.append(label)
    return rows, attrs