import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reduce dataset for faster training
x_train = x_train[:10000]
y_train = y_train[:10000]

# Preprocess
x_train = x_train.reshape(-1,784).astype(np.float32)/255.0
x_test = x_test.reshape(-1,784).astype(np.float32)/255.0

y_train = tf.one_hot(y_train,10)
y_test = tf.one_hot(y_test,10)

input_size = 784
output_size = 10

epochs = 3
batch_size = 128

activation_functions = ["relu","sigmoid"]
hidden_sizes = [64,128]
learning_rates = [0.01]

results = []

def activation(x,name):
    if name=="relu":
        return tf.nn.relu(x)
    elif name=="sigmoid":
        return tf.nn.sigmoid(x)

for act in activation_functions:
    for hidden_units in hidden_sizes:
        for lr in learning_rates:

            print("\nTraining:",act,"Hidden:",hidden_units,"LR:",lr)

            W1 = tf.Variable(tf.random.normal([input_size,hidden_units]))
            b1 = tf.Variable(tf.zeros([hidden_units]))

            W2 = tf.Variable(tf.random.normal([hidden_units,output_size]))
            b2 = tf.Variable(tf.zeros([output_size]))

            optimizer = tf.optimizers.SGD(lr)

            loss_history = []

            for epoch in range(epochs):

                for i in range(0,len(x_train),batch_size):

                    x_batch = x_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]

                    with tf.GradientTape() as tape:

                        z1 = tf.matmul(x_batch,W1)+b1
                        a1 = activation(z1,act)

                        logits = tf.matmul(a1,W2)+b2

                        loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(
                                labels=y_batch,
                                logits=logits
                            )
                        )

                    gradients = tape.gradient(loss,[W1,b1,W2,b2])
                    optimizer.apply_gradients(zip(gradients,[W1,b1,W2,b2]))

                loss_history.append(loss.numpy())
                print("Epoch",epoch+1,"Loss",loss.numpy())

            # Testing
            z1 = tf.matmul(x_test,W1)+b1
            a1 = activation(z1,act)
            test_logits = tf.matmul(a1,W2)+b2

            predictions = tf.argmax(test_logits,axis=1)
            labels = tf.argmax(y_test,axis=1)

            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions,labels),tf.float32)
            )

            print("Accuracy:",accuracy.numpy())

            results.append((act,hidden_units,lr,accuracy.numpy()))

            plt.plot(loss_history,label=f"{act}-H{hidden_units}")

# Loss graph
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy graph
labels = [f"{r[0]} H{r[1]}" for r in results]
accuracies = [r[3] for r in results]

plt.figure()
plt.bar(labels,accuracies)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()