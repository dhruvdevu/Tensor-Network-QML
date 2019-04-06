import tensorflow as tf
import data
import numpy as np

num_epochs = 1
experiment ="params/one_layer"
data_path = data_path = 'data/4'
digits = (0, 1)
val_split = 0.85
batch_size = 25

input = tf.placeholder(tf.float32, shape=[None, 16])
onehot_labels = tf.placeholder(tf.float32, shape = [None, 2])
#Linear layer
# hidden = tf.layers.dense(input, 16, activation=tf.nn.relu)
# output = tf.layers.dense(hidden, 2)
output = tf.layers.dense(input, 2)
loss = tf.losses.softmax_cross_entropy(onehot_labels,output)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
classification_result = tf.argmax(output, axis=1)
train_op = optimizer.minimize(loss)


(train_data, val_data, test_data) = data.get_data(data_path, digits, val_split)
(train_images, train_labels) = train_data
(val_images, val_labels) = val_data
(test_images, test_labels) = test_data
train_images = data.flatten_images(train_images)
val_images = data.flatten_images(val_images)
test_images = data.flatten_images(test_images)
print(train_images.shape, val_images.shape)
with tf.Session() as sess:
    # Here is how you initialize weights of the model according to their
    # Initialization parameters.
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    val_feed = {input: val_images}
    true_val = np.argmax(val_labels, axis=1)

    val_res = sess.run([output], feed_dict=val_feed)
    val_acc = np.sum(true_val.dot(np.argmax(val_res[0], axis=1)))/np.sum(true_val)
    print("Validation accuracy:", val_acc)
    saver.save(sess, experiment)
    for i in range(num_epochs):
        batch_iter = data.batch_generator(train_images, train_labels, batch_size)
        for (images, labels) in batch_iter:
            feed = {input: images, onehot_labels: labels}
            train_loss, op = sess.run([loss, train_op], feed_dict=feed)
        print("Epoch:", i)
        feed = {input: val_images, onehot_labels: val_labels}
        val_res = sess.run([output], feed_dict=val_feed)
        val_acc = np.sum(true_val.dot(np.argmax(val_res[0], axis=1)))/np.sum(true_val)
        print("Validation accuracy:", val_acc)
        saver.save(sess, experiment)


    feed = {input: train_images}
    res = sess.run([output], feed_dict=feed)
    true_class = np.argmax(train_labels, axis=1)
    print("Test accuracy:", np.sum(true_class.dot(np.argmax(res[0], axis=1)))/np.sum(true_class))
