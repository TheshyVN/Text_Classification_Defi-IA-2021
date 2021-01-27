import os
import numpy as np
import pandas as pd
import tensorflow.keras.models as km
import tensorflow as tf
import tensorflow_hub as hub
import time

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print("Mode: ", MODE)

################
### Argument ###
################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--bootstrap', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=3e-5)
args = parser.parse_args()

#####################
### Loading model ###
#####################
if args.bootstrap:
    model_str = 'BERT_bootstrap_'
else:
    model_str = 'BERT_original_'
args_str = "epochs_%d_batch_size_%d_learning_rate_%f" %(args.epochs, args.batch_size,args.learning_rate)
classifier_model = km.load_model("./output/models/" + model_str + args_str+".h5",custom_objects={'KerasLayer':hub.KerasLayer})

#################
### Data test ###
#################
PATH="./data/test/"
test_df2 = pd.read_json(PATH+"test.json")
X_test=tf.convert_to_tensor(test_df2['description'])

##################
### Prediction ###
##################
ts = time.time()
preds=classifier_model.predict(X_test)
te = time.time()
print("Prediction time: ",te-ts)
test_df2["Category"] = np.argmax(preds,axis=1)
output = test_df2[["Id","Category"]]

#################################
### Creating submission file ####
#################################
if not os.path.exists('./output/submission'):
    os.makedirs('./output/submission')
output.to_csv("./output/submission/"+model_str+args_str+".csv", index=False)
print("Submission file saved.")
