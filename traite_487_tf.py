# %%
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

#%%
file_path = 'data/data_asset_choosed/34/subset_by_time_drop_merge/'
file_list = os.listdir(file_path)
keys_data = ["time","distance","average_speed_50"]
keys_prediction = ["fuel"]

input = None
output = None
for file_name in file_list:
    df_i = pd.read_csv(file_path+file_name)
    df_input = df_i[keys_data][1:].convert_objects(convert_numeric=True).as_matrix()
    df_output = df_i[keys_prediction][1:].convert_objects(convert_numeric=True).as_matrix()

    if input == None:
        input = df_input
        output = df_output
    else:
        input = np.concatenate( (input,df_input), axis=0)
        output = np.concatenate( (output,df_output), axis=0)


#%%

[mean_time,mean_distance,mean_speed] = mean_value =np.mean(input, axis=0)
mean_fuel = np.mean(output, axis=0)
input_norm = np.divide(input,mean_value)
output_norm = np.divide(output,mean_fuel)

print(mean_value)
print(mean_fuel)
print("data length : ",len(input_norm),input_norm.shape)
print("data length : ",len(output_norm),output_norm.shape)



#%%
train_data = input_norm[0:3000]
train_label = output_norm[0:3000]
test_data = input_norm[3000:]
test_label = output_norm[3000:]

#%%
batch_size = 1000
input_size = 3
output_size = 1

tf_train_dataset = tf.placeholder(tf.float32, shape=(None,input_size))
tf_train_label   = tf.placeholder(tf.float32, shape=(None,output_size))

size_h1 = 20
size_h2 = 20

w1 = tf.Variable(tf.truncated_normal([3, size_h1]))
b1 = tf.Variable(tf.zeros([size_h1]))
w2 = tf.Variable(tf.truncated_normal([size_h1, size_h2]))
b2 = tf.Variable(tf.zeros([size_h2]))
w3 = tf.Variable(tf.truncated_normal([size_h2, 1]))
b3 = tf.Variable(tf.zeros([1]))

h1 = tf.nn.relu(tf.add(tf.matmul(tf_train_dataset, w1), b1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2))
pred = tf.add(tf.matmul(h2, w3), b3)
loss = tf.reduce_sum(tf.squared_difference(pred,tf_train_label))
optimizer = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#%%

tf_model_path = "data/data_asset_choosed/34/tf_model/model.ckpt"
with tf.Session() as sess:
    sess.run(init)
    num_iters = len(train_data)//batch_size
    num_epochs = 100
    print('variable initialized')
    print('num of iters: ', num_iters)

    offset = 0
    endset = 0+batch_size
    for e in range(num_epochs):
        for i in range(num_iters):

            if endset > len(train_data):
                batch_x = train_data[offset:]
                batch_y = train_label[offset:]

                offset = endset - len(train_data)
                endset = offset + batch_size

                batch_x = np.concatenate( (batch_x,train_data[0:offset]), axis=0)
                batch_y = np.concatenate( (batch_y,train_label[0:offset]), axis=0)


            else:
                batch_x = train_data[offset:endset]
                batch_y = train_label[offset:endset]
                offset = offset + batch_size
                endset = endset + batch_size

            sess.run(optimizer, feed_dict={tf_train_dataset: batch_x, tf_train_label: batch_y})


        lo = sess.run(loss, feed_dict={tf_train_dataset: train_data, tf_train_label: train_label})
        print("loss is :",lo)
    saver.save(sess,tf_model_path)

#%%
tf_model_path = "data/data_asset_choosed/34/tf_model/model.ckpt"
with tf.Session() as sess:
    saver.restore(sess, tf_model_path)
    prediction = sess.run(pred,feed_dict={tf_train_dataset: train_data[0:10]} )
    #print(prediction[0:10])
    #print(train_label[0:10])
    print(np.multiply(prediction[0:10],mean_fuel))
    print(np.multiply(train_label[0:10],mean_fuel))





#%%
