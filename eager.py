# BIOMEDE 517 - Neural Engineering
# Neural Networks for Neural Networks - Understanding Seizure EEG Data
# Main Neural Network using TensorFlow Eager Execution

from __future__ import absolute_import, division, print_function

import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

matplotlib.interactive(True)

# Config eager execution
tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Training Neural Network to classify EEG data into 1 of 5 classes")
print("Kushal Jaligama - BIOMEDE 517 - Engineering Final Project")

# Download the dataset
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
#                                            origin=train_dataset_url)
# train_dataset_fp = tf.keras.utils.get_file(fname="/Users/jollygama/dev/BIOMEDE517/Final Project Data/SpottingSeizure/data.csv")
# train_dataset_fp = "Final Project Data/SpottingSeizure/data.csv"
train_dataset_fp = "Final Project Data/SpottingSeizure/eeg_train.csv"

print("Local copy of the dataset file: {}".format(train_dataset_fp))


# Parse the dataset
def parse_csv(line):
    # example_defaults = [[0.], [0.], [0.], [0.], [0]]  # Set field types
    example_defaults = [[""]]
    example_defaults[1:] = [[1.0]] * 178
    example_defaults.append([1])
    # print(example_defaults)
    # print(len(example_defaults))
    parsed_line = tf.decode_csv(line, example_defaults)
    print("Number of samples in one training example", len(parsed_line))
    # First 4 fields are features, combine into single tensor
    # features = tf.reshape(parsed_line[:-1], shape=(4,))

    # Each input example has 178 samples of eeg data
    features = tf.reshape(parsed_line[1:-1], shape=(178,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

# Create training tf.data.Dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
# Skip the first row which is the header - took this out in eeg_train.csv and eeg_test.csv
# train_dataset = train_dataset.skip(1)
# Parse each row
train_dataset = train_dataset.map(parse_csv)
# Randomize the data
train_dataset = train_dataset.shuffle(buffer_size=1000)

# Split the dataset into number of samples per training batch (rows)
# Argument represents elements in one input (32 originally)
batch_size = 50
train_dataset = train_dataset.batch(batch_size)
print("Data set randomized and split into {} batches".format(batch_size))

# View single example entry from batch
features, label = tfe.Iterator(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

# Select the type of model - tf.keras is the preferred way
model = tf.keras.Sequential([
    # The input layer's shape is required
    # Activation function is loosely based on how brain neurons are connected
    tf.keras.layers.Dense(20, activation="relu", input_shape=(178,)),
    # Hidden/Middle Layer
    tf.keras.layers.Dense(20, activation="relu"),
    # Output layer size, this is equivalent to number of classes/labels
    tf.keras.layers.Dense(5)
])
print("Neural Network Model Initialized (Keras)")


# Train the model - stage where model is gradually optimized,
# learns the structure of the dataset to be able to make predictions about
# unseen data

# Loss function measures error of model's predictions from desired label
# Want to minimize this value
def loss(model, x, y):
    y_predict = model(x)
    # print(y)
    labels = tf.one_hot(y - 1, 5)
    # print(labels)
    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=y_predict)

# Use loss function and tfe.GradientTape to record operations that compute
# gradients used to optimize model
def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

# Create an optimizer - standard gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Training loop for model - feed dataset exampls into model to help it make
# better predictions

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 10

print("Beginning Training: {} epochs", num_epochs)
# Iterate through each epoch (one pass through the entire dataset)
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Within an epoch, iterate over each batch in the training dataset
    # and grab the features and associated labels
    current_batch = 0 # Use this to keep track of which batch is being used to train the network
    for x, y in tfe.Iterator(train_dataset):
        # This loop trains the network over the number of samples in each batch
        # Using the example's features, make a prediction and compare it with
        # the label. Measure inaccuracy of prediction and use to calculate
        # the model's loss and gradients.

        # Optimize the model
        grads = grad(model, x, y)
        # Use optimizer to update model's variables
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.
                                  get_or_create_global_step())

        # Track progress - use for visualization
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y - 1)
        # print("After batch: {} --- Accuracy: {}".format(current_batch, epoch_accuracy.result()))
        current_batch += 1

    # end epoch - repeat inner for loop for each epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # if epoch % 50 == 0:
    print("Epoch: {:03d}: Loss: \
        {:.3f}, Accuracy: {:.3%}".format(epoch,
                                         epoch_loss_avg.result(),
                                         epoch_accuracy.result()))

# Visualize the loss function over time
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()

# Evaluate Model Effectiveness

# # Get the test dataset
# test_url = "http://download.tensorflow.org/data/iris_test.csv"

# test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
#                                   origin=test_url)

test_dataset = tf.data.TextLineDataset("Final Project Data/SpottingSeizure/eeg_test.csv")
# test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(batch_size)           # use the same batch size as the training set

# # Evaluate the model on the test dataset
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in tfe.Iterator(test_dataset):
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y - 1)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
