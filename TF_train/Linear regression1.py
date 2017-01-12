import tensorflow as tf
hello = tf.constant('hello')
x_data = [1, 2, 3]
y_data = [1, 2, 3]

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# Variable로 선언해주어야 업데이트 가능, -1 ~ 1 사이의 랜덤 값 부여

hypothesis = w * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis-y_data))
# cost function의 공식

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(w), sess.run(b))