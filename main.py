import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
adder_node = a + b
add_and_tripple = adder_node * 3.

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
	sess.run(train, {x: [1,2,3,4], y: [0, -1, -2, -3]})
	
print(sess.run([W, b]))