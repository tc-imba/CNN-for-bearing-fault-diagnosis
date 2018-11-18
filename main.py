import tensorflow as tf
import numpy as np

batch_size = 4096
fault_num = 5
train_count = 10
test_count = 10

train_data = np.zeros((fault_num * train_count, batch_size), dtype=np.float64)
train_labels = np.zeros((fault_num * train_count, 1, fault_num), dtype=np.float64)
train_counter = 0
test_data = np.zeros((fault_num * test_count, batch_size), dtype=np.float64)
test_labels = np.zeros((fault_num * test_count, 1, fault_num), dtype=np.float64)
test_counter = 0

for i in range(0, fault_num):
    filename = 'data/422_1_%d.tdms01.txt' % i
    with open(filename) as f:
        for j, line in enumerate(f.readlines()):
            if j / batch_size < train_count:
                if j % batch_size == 0:
                    train_labels[train_counter][0][i] = 1.
                    train_counter += 1
                train_data[train_counter - 1][j % batch_size] = np.float64(line)
            elif j / batch_size < train_count + test_count:
                if j % batch_size == 0:
                    test_labels[test_counter][0][i] = 1.
                    test_counter += 1
                test_data[test_counter - 1][j % batch_size] = np.float64(line)


train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

print(train_data.shape)


def shifted_relu(x):
    return tf.maximum(-1., x)


model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=12, kernel_size=3, padding='same', input_shape=(4096, 1)),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=12, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=24, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=24, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=48, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=48, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=96, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=96, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    tf.keras.layers.Activation(activation=shifted_relu),
    tf.keras.layers.MaxPool1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=fault_num, kernel_size=1, padding='same'),
    tf.keras.layers.AveragePooling1D(pool_size=32),
    tf.keras.layers.Softmax()
])

model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse', metrics=['mae'])

model.summary()

model.fit(train_data, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(test_loss, test_acc)
