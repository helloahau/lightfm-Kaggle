import os
import pandas as pd
import create_validation as val
import preprocessing_submission as prep_sub
import preprocessing_validation as prep_val
import translate as tr
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import pickle
from datetime import date
import calendar
import scipy.io as spi
import scipy.sparse as sps
from sklearn.feature_extraction import DictVectorizer


assert (
    os.path.isfile("../Data/Data_translated/coupon_detail_train_translated.csv"))
assert (
    os.path.isfile("../Data/Data_translated/coupon_visit_train_translated.csv"))
assert (os.path.isfile("../Data/Data_translated/coupon_list_train_translated.csv"))
assert (os.path.isfile("../Data/Data_translated/coupon_list_test_translated.csv"))
assert (os.path.isfile("../Data/Data_translated/user_list_translated.csv"))

df = pd.read_csv("../Data/Data_translated/user_list_translated.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df.iloc[:5, :6], flush=True)


#val.create_validation_set([2012, 6, 17], [2012, 6, 23], "week52")

assert (os.path.isfile(
    "../Data/Validation/week52/coupon_detail_train_validation_week52.csv"))
assert (os.path.isfile(
    "../Data/Validation/week52/coupon_visit_train_validation_week52.csv"))
assert (os.path.isfile(
    "../Data/Validation/week52/coupon_list_train_validation_week52.csv"))
assert (os.path.isfile(
    "../Data/Validation/week52/coupon_list_test_validation_week52.csv"))
assert (
    os.path.isfile("../Data/Validation/week52/user_list_validation_week52.csv"))