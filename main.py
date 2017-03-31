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

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
adder_node = a + b
add_and_tripple = adder_node * 3.

#print(node1, node2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#print(sess.run([node1, node2]))
#print("node3:", node3)
#print("sess.run(node3):", sess.run(node3))
#print(sess.run(adder_node, {a: 3, b: 4.5}))
#print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
#print(sess.run(add_and_tripple, {a: 3, b: 4.5}))
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))