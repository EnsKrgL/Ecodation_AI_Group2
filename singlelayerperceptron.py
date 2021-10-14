from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

import tensorflow as tf
import matplotlib.pyplot as plt

#parametreler
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

x = tf.placeholder("float", [None, 784]) #mnist veri imajları
#28*28 = 784
y = tf.placeholder("float", [None, 10]) #0-9 karakter sınıfı sayısı

#model yaratma

#model ağırlıklarını belirleme
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#modelimizi inşa etme 
activation = tf.nn.softmax(tf.matmul(x, W) + b)

#cross_entropy kullanarak hatayı minimize etme
cross_entropy = y * tf.log(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Çizim ayarları
avg_set = []
epoch_set = []

#değişkenlerin ilk atamalarını yapma işlemi
init = tf.initialize_all_variables()

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
    
    print("Training fazı tamamlandı")
    
    plt.plot(epoch_set, avg_set, 'o', label = 'Logistic Regression Training phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    #Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    #accuracy (keskinlik) hesabı
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Model accuracy: ", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
