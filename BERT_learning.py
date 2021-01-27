import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer
tf.get_logger().setLevel('ERROR')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle
import time
import os

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

###############
### Dataset ###
###############
PATH="./data/train/"
train_label = pd.read_csv(PATH+"train_label.csv")
train_df2 = pd.read_json(PATH+"train.json")
train_df2['Category'] = train_label['Category'].values
## with bootstrap
if args.bootstrap:
    categories = np.unique(train_label['Category'].values)
    gender_count = []
    gender_ratio = []
    sub_dfs = []
    for i in categories:
        sub_df = train_df2[train_df2['Category']==i]
        sub_dfs.append(sub_df)
        unique, count = np.unique(sub_df['gender'].values, return_counts=True)
        gender_count.append(np.array(count))
        gender_ratio.append(np.max(count)/np.min(count))
    class_count = np.sum(gender_count, axis=1)
    class_ratio = class_count/np.sum(class_count)
    new_class_count = np.copy(class_count)
    # bootstrap goal for balanced genders
    for i, ratio in enumerate(gender_ratio):
        if ratio>2:
            new_class_count[i] = 2*np.max(gender_count[i])
    # bootstrap goal for balanced classes, here we want each class to contribute at
    # least 2% of the training dataset
    continu = True
    new_class_ratio = new_class_count/np.sum(new_class_count)
    while continu:
        which_to_bootstrap = new_class_ratio<0.02
        new_class_count += which_to_bootstrap*1
        continu = np.sum(which_to_bootstrap)
        new_class_ratio = new_class_count/np.sum(new_class_count)
        
    import random as rd
    rd.seed(42)
    new_sub_dfs = []
    for i, count in enumerate(gender_count):
        sub_df = sub_dfs[i]
        male_sub_df = sub_df[sub_df['gender']=='M']
        female_sub_df = sub_df[sub_df['gender']=='F']
        n_male = len(male_sub_df)
        n_female = len(female_sub_df)
        bootstrap_idx_male = np.random.randint(0,n_male,max(0,int(new_class_count[i]/2)-count[1]))
        bootstrap_idx_female = np.random.randint(0,n_female,max(0,int(new_class_count[i]/2)-count[0]))
        sub_df = sub_df.append(male_sub_df.iloc[bootstrap_idx_male.tolist()])
        sub_df = sub_df.append(female_sub_df.iloc[bootstrap_idx_female.tolist()]) 
        new_sub_dfs.append(sub_df) 

    train_bootstrapped = pd.concat(new_sub_dfs,axis=0)
    train_bootstrapped = train_bootstrapped.sample(frac=1).reset_index(drop=True)
    train_bootstrapped.drop(['gender'], axis=1, inplace=True)

    X_train, X_val = train_test_split(train_bootstrapped, test_size=0.2, stratify=train_bootstrapped['Category'], random_state=42)
## without bootstrap
else:
    X_train, X_val = train_test_split(train_df2, test_size=0.2, stratify=train_df2['Category'], random_state=42)

X_train2=tf.convert_to_tensor(X_train['description'])
y_train2=tf.convert_to_tensor(X_train['Category'])
X_val2=tf.convert_to_tensor(X_val['description'])
y_val2=tf.convert_to_tensor(X_val['Category'])

##########################################
### Loading models from TensorFlow Hub ###
##########################################
#@title Choose a BERT model to fine-tune

bert_model_name = 'bert_en_cased_L-12_H-768_A-12'  #@param ["bert_en_uncased_L-12_H-768_A-12", "bert_en_cased_L-12_H-768_A-12", "bert_multi_cased_L-12_H-768_A-12", "small_bert/bert_en_uncased_L-2_H-128_A-2", "small_bert/bert_en_uncased_L-2_H-256_A-4", "small_bert/bert_en_uncased_L-2_H-512_A-8", "small_bert/bert_en_uncased_L-2_H-768_A-12", "small_bert/bert_en_uncased_L-4_H-128_A-2", "small_bert/bert_en_uncased_L-4_H-256_A-4", "small_bert/bert_en_uncased_L-4_H-512_A-8", "small_bert/bert_en_uncased_L-4_H-768_A-12", "small_bert/bert_en_uncased_L-6_H-128_A-2", "small_bert/bert_en_uncased_L-6_H-256_A-4", "small_bert/bert_en_uncased_L-6_H-512_A-8", "small_bert/bert_en_uncased_L-6_H-768_A-12", "small_bert/bert_en_uncased_L-8_H-128_A-2", "small_bert/bert_en_uncased_L-8_H-256_A-4", "small_bert/bert_en_uncased_L-8_H-512_A-8", "small_bert/bert_en_uncased_L-8_H-768_A-12", "small_bert/bert_en_uncased_L-10_H-128_A-2", "small_bert/bert_en_uncased_L-10_H-256_A-4", "small_bert/bert_en_uncased_L-10_H-512_A-8", "small_bert/bert_en_uncased_L-10_H-768_A-12", "small_bert/bert_en_uncased_L-12_H-128_A-2", "small_bert/bert_en_uncased_L-12_H-256_A-4", "small_bert/bert_en_uncased_L-12_H-512_A-8", "small_bert/bert_en_uncased_L-12_H-768_A-12", "albert_en_base", "electra_small", "electra_base", "experts_pubmed", "experts_wiki_books", "talking-heads_base"]

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/2',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

####################
### Define model ###
####################
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net= tf.keras.layers.Dense(64, activation='relu')(net)
  net = tf.keras.layers.Dropout(0.1)(net)
  net= tf.keras.layers.Dense(32, activation='relu')(net)

  net = tf.keras.layers.Dense(28, activation='softmax', name='classifier')(net)
  return tf.keras.Model(text_input, net)
  
classifier_model = build_classifier_model()

#######################
### Model training ####
#######################
## Optimizer
seed = 42
steps_per_epoch = int( np.ceil(X_train2.shape[0] / args.batch_size) )
num_train_steps = steps_per_epoch * args.epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = args.learning_rate
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

## Loading the BERT model and training
classifier_model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.sparse_categorical_crossentropy,
                         metrics='accuracy')
print(f'Training model with {tfhub_handle_encoder}')
ts = time.time()
history = classifier_model.fit(X_train2,y_train2,
                               validation_data=(X_val2,y_val2), batch_size=args.batch_size,
                               epochs=args.epochs)
te = time.time()
t_learning = te - ts
print("Learning time: ",t_learning)

#############################
### Save model and result ###
#############################
if args.bootstrap:
    model_str = 'BERT_bootstrap_'
else:
    model_str = 'BERT_original_'
args_str = "epochs_%d_batch_size_%d_learning_rate_%f" %(args.epochs, args.batch_size,args.learning_rate)
if not os.path.exists('./output/models'):
    os.makedirs('./output/models')
classifier_model.save("./output/models/" + model_str + args_str+".h5", include_optimizer=False)
print("Model saved.")

if not os.path.exists('./output/results'):
    os.makedirs('./output/results')
his = history.history
results = vars(args)
results.update({"t_learning": t_learning, "loss_train": his['loss'][-1],"accuracy_train": his['accuracy'][-1],
                 "loss_val": his['val_loss'][-1], "accuracy_val" : his['val_accuracy'][-1]})
pickle.dump(results, open("./output/results/" + model_str + args_str  + ".pkl", "wb"))
print("Result saved.")

##################
### Prediction ### this part will be splited into another file (BERT_prediction.py) in future when the problem of tensorflow text could be resovled 
##################
## Data test
PATH="./data/test/"
test_df2 = pd.read_json(PATH+"test.json")
X_test=tf.convert_to_tensor(test_df2['description'])
## Prediction
ts = time.time()
preds=classifier_model.predict(X_test)
te = time.time()
print("Prediction time: ",te-ts)
test_df2["Category"] = np.argmax(preds,axis=1)
output = test_df2[["Id","Category"]]
## Creating submission file
if not os.path.exists('./output/submission'):
    os.makedirs('./output/submission')
output.to_csv("./output/submission/"+model_str+args_str+".csv", index=False)
print("Submission file saved.")
###########################
### End prediction part	###
###########################
