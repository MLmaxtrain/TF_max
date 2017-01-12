import tensorflow as tf
hello = tf.constant('hello')
x_data = [1, 2, 3]
y_data = [1, 2, 3]

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# Variable로 선언해주어야 업데이트 가능, -1 ~ 1 사이의 랜덤 값 부여
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = w * X + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# cost function의 공식

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print (step, sess.run(cost,feed_dict={X:x_data, Y:y_data}), sess.run(w), sess.run(b))

print (sess.run(hypothesis, feed_dict={X:2.5}))