import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw') + os.sep
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed') + os.sep
SQL_DIR = os.path.join(BASE_DIR, 'sql') + os.sep
MODEL_DIR = os.path.join(BASE_DIR, 'models') + os.sep
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks') + os.sep

TARGET = 'TARGET'
ID_COL = 'SK_ID_CURR'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

if __name__ == '__main__':
    print(f"BASE_DIR:      {BASE_DIR}")
    print(f"DATA_DIR:      {DATA_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"MODEL_DIR:     {MODEL_DIR}")