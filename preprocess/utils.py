import numpy as np
import pandas as pd
import os
import tensorflow.compat.v1 as tf

def read_data(source_dir, CONTS, CATES, name='train'):
    file = os.path.join(source_dir, name + '.txt')
    cate_cols = ["cate_{}".format(i+1) for i in range(CATES)]
    cont_cols = ["cont_{}".format(i+1) for i in range(CONTS)]
    cont_cate_cols = cont_cols + cate_cols
    columns = ['click'] + cont_cate_cols
    df = pd.read_csv(file, sep='\t', names=columns, header=None, low_memory=False)
    df_y = df['click']
    df_X = df[cont_cate_cols]

    return df_X.to_numpy(), df_y.to_numpy()

def generate_cate_dict(target_dir, train_X, cate_cols, cont_cols, X=20):
    shifts = []
    values = []
    cont_cate_cols = cont_cols + cate_cols
    for i, cat in enumerate(cate_cols):
        index = i + len(cont_cols)
        nan_index = index
        a = np.array([str(x1) for x1 in train_X[:,index]])
        value, count = np.unique(a, return_counts=True)
        new_value = list(set([value[j] if count[j] >= X else nan_index for j, _ in enumerate(value)]))
        if index in new_value:
            new_value.remove(nan_index)
        print(cat, new_value[0:10])
        shifts.append(len(new_value))
        values.append(new_value)
    
    # Print nan index for each categorical feature
    for i, cat in enumerate(cate_cols):
        print(cat, 'nan' in values[i])

    # Generate dictionary
    dict_dir = os.path.join(target_dir, 'X_' + str(X), 'dict_cate')
    os.makedirs(dict_dir, exist_ok=True)
    for i, feat_number in enumerate(shifts):
        index = np.arange(feat_number).astype('int32') + np.sum(shifts[0:i]).astype('int32') + int(len(cont_cate_cols))
        result = np.column_stack([index, values[i]])
        cat_file = os.path.join(dict_dir, "train_out_cat_" + str(i) + ".h5")
        np.save(cat_file, result)
    print("Finish generate dictionary")

def generate_comb_dict(dict_dir, train_X, selected_pairs, Y=20):
    shifts = []
    values = []
    for index, (i, j) in enumerate(selected_pairs):
        a = np.array([str(x1) + '-' + str(x2) for x1, x2 in zip(train_X[:,i], train_X[:,j])])
        value, count = np.unique(a, return_counts=True)
        new_value = list(set([value[i] if count[i] >= Y else index for i, _ in enumerate(value)]))
        if index in new_value:
            new_value.remove(index)
        print(i,j,new_value[0:10])
        shifts.append(len(new_value))
        values.append(new_value)

    # Generate dictionary
    os.makedirs(dict_dir, exist_ok=True)
    for i, shift in enumerate(shifts):
        index = np.arange(shift).astype('int32') + np.sum(shifts[0:i]).astype('int32') + int(len(selected_pairs))
        result = np.column_stack([index, values[i]])
        comb_file = os.path.join(dict_dir, 'train_out_comb_' + str(i) + '_Y_' + str(Y) + '.h5')
        np.save(comb_file, result)
    print("Finish generate combinational dictionary")

def process_cont(df_X, cont_cols):
    for index, cont in enumerate(cont_cols):
        a = np.float32(df_X[:,index])
        a_min = np.nanmin(a)
        a_max = np.nanmax(a)
        b = (a - a_min)/(a_max - a_min)
        b[np.isnan(b)] = 0.
        b = b.reshape(-1,1)
        if index == 0:
            new_X = b
            print(index, new_X.shape, end='\t')
        else:
            new_X = np.append(new_X, b, axis=-1)
            print(index, new_X.shape, end='\t')
    return new_X

def process_cate(df_X, cont_cols, cate_cols, dict_dir):
    for i, cate in enumerate(cate_cols):
        index = len(cont_cols) + i
        mapping_dict = np.load(os.path.join(dict_dir, "train_out_cat_" + str(i) + ".h5.npy"))
        my_dict = dict(zip(mapping_dict[:,1], mapping_dict[:,0]))
        a = np.array([str(x1) for x1 in df_X[:,index]])
        b = np.vectorize(my_dict.get)(a)
        b = np.where(b=='None', index, b)
        b = np.where(b==None, index, b)
        b = b.astype('int32').reshape(-1,1)
        if i == 0:
            new_X = b
            print(i, new_X.shape, end='\t')
        else:
            new_X = np.append(new_X, b, axis=-1)
            print(i, new_X.shape, end='\t')
    return new_X

def process_comb(df_X, dict_dir, selected_pairs, Y=20):
    for index, (i,j) in enumerate(selected_pairs):
        mapping_dict = np.load(os.path.join(dict_dir, 'train_out_comb_' + str(i) + '_Y_' + str(Y) + '.h5.npy'))
        my_dict = dict(zip(mapping_dict[:,1], mapping_dict[:,0]))
        comb = "comb_" + str(i+1) + "_" + str(j+1)
        a = np.array([str(x1) + '-' + str(x2) for x1, x2 in zip(df_X[:,i], df_X[:,j])])
        b = np.vectorize(my_dict.get)(a)
        b = np.where(b==None, index, b)
        b = np.where(b=='None', index, b)
        b = np.where(['None' in str(xb) for xb in b], index, b)
        b = b.astype('int32').reshape(-1,1)
        if index == 0:
            comb_X = np.array(b).reshape(-1,1)
            print(index, comb_X.shape, end='\t')
        else:
            comb_X = np.append(comb_X, b, axis=1)
            print(index, comb_X.shape, end='\t')
    comb_X = comb_X.astype('int32')
    return comb_X

def write_to_tfrecord(df_X, df_y, orig_dir, CONTS, CATES, COMBS, name='train', partnum=2000000, line_size=1000):
    startindex = 0
    total_len = df_X.shape[0]
    part = 0
    os.makedirs(orig_dir, exist_ok=True)
    for startindex in np.arange(0, total_len, partnum):
        if startindex + partnum >= total_len:
            endindex = total_len
        else:
            endindex = startindex + partnum
            endindex -= endindex % line_size
        save_file_path = os.path.join(orig_dir, name + '_part_' + str(part) + '.tfrecord')
        writer = tf.python_io.TFRecordWriter(save_file_path)
        chunk_label = np.asarray(df_y[startindex:endindex], dtype=np.float32)
        chunk_conts = np.asarray(df_X[startindex:endindex,0:CONTS], dtype=np.float32)
        chunk_cates = np.asarray(df_X[startindex:endindex,CONTS:CONTS+CATES], dtype=np.int64)
        chunk_combs = np.asarray(df_X[startindex:endindex,CONTS+CATES:CONTS+CATES+COMBS], dtype=np.int64)
        for idx in range(0, endindex-startindex, line_size):
            if idx + line_size > endindex-startindex:
                continue
            line_label = chunk_label[idx : idx + line_size].flatten().tolist()
            line_conts = chunk_conts[idx : idx + line_size].flatten().tolist()
            line_cates = chunk_cates[idx : idx + line_size].flatten().tolist()
            line_combs = chunk_combs[idx : idx + line_size].flatten().tolist()
            assert len(line_label) == line_size * 1 , "line_label.size: {}".format(len(line_label))
            assert len(line_conts) == line_size * CONTS, "line_conts.size: {}".format(len(line_conts))
            assert len(line_cates) == line_size * CATES, "line_cates.size: {}".format(len(line_cates))
            assert len(line_combs) == line_size * COMBS, "line_combs.size: {}".format(len(line_combs))

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=line_label)),
                "feat_conts": tf.train.Feature(float_list=tf.train.FloatList(value=line_conts)),
                "feat_cates": tf.train.Feature(int64_list=tf.train.Int64List(value=line_cates)),
                "feat_combs": tf.train.Feature(int64_list=tf.train.Int64List(value=line_combs))
            }))
            writer.write(example.SerializeToString())
        writer.close()
        part += 1
