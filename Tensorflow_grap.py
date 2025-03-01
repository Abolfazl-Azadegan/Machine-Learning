import tensorflow as tf
# Define a TensorFlow summary writer
log_dir = "logs/"
writer = tf.summary.create_file_writer(log_dir)

# Create a simple computation
x = tf.constant(2.0)
y = tf.constant(3.0)
z = x * y  # Some operation

# Log the computation graph
tf.summary.trace_on(graph=True, profiler=True)
with writer.as_default():
    tf.summary.trace_export(name="my_graph", step=0)