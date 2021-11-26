import tensorflow as tf
import glob
import torch

SAMPLES = 1000
CONTS = 13
CATES = 26

@tf.autograph.experimental.do_not_convert
def read_data(raw_rec):
    feature_des = {
        "label": tf.io.FixedLenFeature([SAMPLES], tf.float32),
        "feat_conts": tf.io.FixedLenFeature([SAMPLES * CONTS], tf.float32),
        "feat_cates": tf.io.FixedLenFeature([SAMPLES * CATES], tf.int64),
    }
    example = tf.io.parse_single_example(raw_rec, feature_des)
    inputs = {}
    conts = tf.reshape(example["feat_conts"], [SAMPLES, CONTS])
    cates = tf.reshape(example["feat_cates"], [SAMPLES, CATES])
    inputs["feat_cates"] = cates
    inputs["feat_conts"] = conts
    return inputs, example["label"]

@tf.autograph.experimental.do_not_convert
def read_data_comb(raw_rec):
    feature_des = {
        "label": tf.io.FixedLenFeature([SAMPLES], tf.float32),
        "feat_conts": tf.io.FixedLenFeature([SAMPLES * CONTS], tf.float32),
        "feat_cates": tf.io.FixedLenFeature([SAMPLES * CATES], tf.int64),
        "feat_combs": tf.io.FixedLenFeature([SAMPLES * COMBS], tf.int64),
    }
    example = tf.io.parse_single_example(raw_rec, feature_des)
    inputs = {}
    conts = tf.reshape(example["feat_conts"], [SAMPLES, CONTS])
    cates = tf.reshape(example["feat_cates"], [SAMPLES, CATES])
    combs = tf.reshape(example["feat_combs"], [SAMPLES, COMBS])
    inputs["feat_conts"] = conts
    inputs["feat_cates"] = cates
    inputs['feat_combs'] = combs
    return inputs, example["label"]

@tf.autograph.experimental.do_not_convert
def get_data(folder, name="train", bsize=2, use_comb=False, comb_field=0):
    files = glob.glob(folder + '/' + name + "*.tfrecord")
    if use_comb:
        global COMBS 
        COMBS = comb_field
        ds = tf.data.TFRecordDataset(files).map(read_data_comb,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.data.TFRecordDataset(files).map(read_data,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    
    for x,y in ds:
        x = { k: torch.from_numpy(v.numpy()) for k,v in x.items() }
        y = y.numpy()
        yield x, torch.from_numpy(y)

