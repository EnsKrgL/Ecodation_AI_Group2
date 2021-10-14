#multilayer perceptron
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

import tensorflow as tf
import matplotlib.pyplot as plt

#parametreler
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

n_hidden_1 = 256 #ilk katmana ait öznitelik (feature) sayısı
n_hidden_2 = 256 #ikinci katmana ait öznitelik (feature) sayısı
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input]) #mnist veri imajları
#28*28 = 784
y = tf.placeholder("float", [None, n_classes]) #0-9 karakter sınıfı sayısı

#1. katmandaki ağırlıklar 
h = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
#bias katman 1
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))
#katman 1
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,h), bias_layer_1))

#2. katmandaki ağırlıklar 
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
#bias katman 2
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))
#katman 2
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w), bias_layer_2))

#çıkış katmandaki ağırlıklar 
output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
#bias çıkıl katmanı
bias_output = tf.Variable(tf.random_normal([n_classes]))
#çıkış katmanı
output_layer = tf.matmul(layer_2, output) + bias_output

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output_layer, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Çizim ayarları
avg_set = []
epoch_set = []

#değişkenlerin ilk atamalarını yapma işlemi
init = tf.global_variables_initializer()

#grafiği yerleştirme
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            #eğitimi batch verisi ile uyumlu hale getirme işlemi
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            
            #ortalama kayıbı (loss) hesaplama
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch


        #her bir epoch adımı için log'ları gösterme
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    
    print("Eğitim fazı tamamlandı")
    
    plt.plot(epoch_set, avg_set, 'o', label = 'MLP eğitim fazı')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    #Test model
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    #accuracy (keskinlik) hesabı
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Model accuracy: ", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
