import pickle
import scipy.sparse as sps
import numpy as np
import pandas as pd
import scipy.io as spi

loaded_model = pickle.load(open("../Data/Model/model.pickle", 'rb'))
week_ID = 'week52'
uf = spi.mmread(
    "../Data/Validation/%s/user_feat_mtrx_%s.mtx" %
    (week_ID, week_ID))
itef = spi.mmread(
    "../Data/Validation/%s/test_item_feat_mtrx_%s.mtx" %
    (week_ID, week_ID))

cplte = pd.read_csv(
    "../Data/Validation/" +
    week_ID +
    "/coupon_list_test_validation_" +
    week_ID +
    ".csv")
ulist = pd.read_csv(
    "../Data/Validation/" +
    week_ID +
    "/user_list_validation_" +
    week_ID +
    ".csv")
list_coupon = cplte["COUPON_ID_hash"].values
list_user = ulist["USER_ID_hash"].values

test = sps.csr_matrix(
    (len(list_user),
     len(list_coupon)),
    dtype=np.int32)
no_items = test.shape[1]
pid_array = np.arange(no_items, dtype=np.int32)
for user_id, row in enumerate(test):
    print("\rProcessing user " + str(user_id) + "/ " + str(len(list_user)))
    uid_array = np.empty(no_items, dtype=np.int32)
    uid_array.fill(user_id)
    predictions = loaded_model.predict(
        uid_array,
        pid_array,
        user_features=uf,
        item_features=itef,
        num_threads=4)
    print(predictions)
    break
