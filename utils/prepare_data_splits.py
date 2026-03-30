import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

CATEGORICAL_COLS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

NUMERICAL_COLS = [
    'age', 'fnlwgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

# Non-IID split: 5 clients partitioned by education-num (1–16)
# education-num mapping:
#   1=Preschool, 2=1st-4th, 3=5th-6th, 4=7th-8th
#   5=9th, 6=10th, 7=11th, 8=12th
#   9=HS-grad, 10=Some-college
#   11=Assoc-voc, 12=Assoc-acdm, 13=Bachelors
#   14=Masters, 15=Prof-school, 16=Doctorate
EDUCATION_GROUPS = {
    0: list(range(1, 5)),    # Elementary  (edu-num 1–4)
    1: list(range(5, 9)),    # Some HS     (edu-num 5–8)
    2: [9, 10],              # HS / Some College
    3: [11, 12, 13],         # Assoc / Bachelors
    4: [14, 15, 16],         # Masters / Prof / Doctorate
}

CLIENT_LABELS = [
    "Elementary\n(edu-num 1–4)",
    "Some High School\n(edu-num 5–8)",
    "HS Grad / Some College\n(edu-num 9–10)",
    "Assoc / Bachelors\n(edu-num 11–13)",
    "Masters / Prof / Doctorate\n(edu-num 14–16)",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adult_dataset(data_dir='data/adult', split='train'):
    """
    Load Adult Census dataset from local files.

    Parameters
    ----------
    data_dir : str
        Path to the adult data directory (default: 'data/adult')
    split : str
        Which dataset to load: 'train' or 'test' (default: 'train')

    Returns
    -------
    X : pd.DataFrame  – features (excluding target)
    y : pd.Series     – target variable ('>50K' or '<=50K')
    """
    if split == 'train':
        file_path = os.path.join(data_dir, 'adult.data')
        skiprows = 0
    elif split == 'test':
        file_path = os.path.join(data_dir, 'adult.test')
        skiprows = 1   # skip the comment line in the test file
    else:
        raise ValueError("split must be 'train' or 'test'")

    df = pd.read_csv(
        file_path,
        names=COLUMN_NAMES,
        sep=', ',
        engine='python',
        na_values='?',
        skiprows=skiprows
    )

    X = df.drop('income', axis=1)
    y = df['income'].str.strip('.')   # remove trailing '.' present in test file

    return X, y


def load_adult_both_splits(data_dir='data/adult'):
    """
    Load both training and test splits of the Adult Census dataset.

    Returns
    -------
    X_train, y_train, X_test, y_test : (pd.DataFrame, pd.Series) × 2
    """
    X_train, y_train = load_adult_dataset(data_dir, split='train')
    X_test, y_test = load_adult_dataset(data_dir, split='test')
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Preprocessing + non-IID split pipeline
# ---------------------------------------------------------------------------

def preprocess_and_split(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test_raw: pd.Series,
):
    """
    Full preprocessing pipeline + non-IID client splits.

    Steps
    -----
    1. Drop rows with missing values.
    2. Encode target as binary (>50K → 1, <=50K → 0).
    3. One-hot encode categorical features (train + test together for alignment).
    4. Standardise numerical features (fit on training set only).
    5. Partition training data into 5 non-IID client splits by education level.

    Parameters
    ----------
    X_train_raw, y_train_raw : raw training features and labels
    X_test_raw,  y_test_raw  : raw test features and labels
    Returns
    -------
    X_train      : np.ndarray  shape (N_train, F)
    y_train      : np.ndarray  shape (N_train,)  binary
    X_test       : np.ndarray  shape (N_test,  F)
    y_test       : np.ndarray  shape (N_test,)   binary
    client_data  : list of (X, y) tuples, one per client
    feature_names: list[str] – column names after encoding
    client_info  : dict – per-client metadata (n_samples, %high-income, label)
    """
    # --- Step 1: drop missing values ----------------------------------------
    train_df = X_train_raw.copy()
    train_df['__y__'] = y_train_raw.values
    train_df.dropna(inplace=True)

    test_df = X_test_raw.copy()
    test_df['__y__'] = y_test_raw.values
    test_df.dropna(inplace=True)

    # --- Step 2: binary targets ---------------------------------------------
    y_train = (train_df.pop('__y__') == '>50K').astype(np.float64).values
    y_test  = (test_df.pop('__y__')  == '>50K').astype(np.float64).values

    # Save education-num *before* encoding for client partitioning
    edu_num_train = train_df['education-num'].values.copy()

    # --- Step 3: one-hot encode (combine so columns stay aligned) -----------
    n_train   = len(train_df)
    combined  = pd.concat([train_df, test_df], ignore_index=True)
    encoded   = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dtype=np.float64)

    feature_names = encoded.columns.tolist()
    X_all         = encoded.values.astype(np.float64)
    X_train_enc   = X_all[:n_train].copy()
    X_test_enc    = X_all[n_train:].copy()

    # --- Step 4: standardise numerical features (train stats only) ----------
    num_idx = [i for i, c in enumerate(feature_names) if c in NUMERICAL_COLS]
    mean = X_train_enc[:, num_idx].mean(axis=0)
    std  = X_train_enc[:, num_idx].std(axis=0) + 1e-8

    X_train_enc[:, num_idx] = (X_train_enc[:, num_idx] - mean) / std
    X_test_enc[:, num_idx]  = (X_test_enc[:, num_idx]  - mean) / std

    # --- Step 5: non-IID client splits by education level -------------------
    client_data = []
    client_info = {}

    for cid, edu_nums in EDUCATION_GROUPS.items():
        mask = np.isin(edu_num_train, edu_nums)
        X_c, y_c = X_train_enc[mask], y_train[mask]
        client_data.append((X_c, y_c))
        client_info[cid] = {
            'n_samples':       int(mask.sum()),
            'high_income_pct': float(y_c.mean() * 100) if len(y_c) else 0.0,
            'edu_nums':        edu_nums,
            'label':           CLIENT_LABELS[cid],
        }

    return X_train_enc, y_train, X_test_enc, y_test, client_data, feature_names, client_info
