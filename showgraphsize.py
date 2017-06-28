import tensorflow as tf
import gardendata
NUM_CLASSES = len(gardendata.labels)

import gardengraph
graph = gardengraph.cnn(NUM_CLASSES)

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
sess.run( tf.local_variables_initializer() )


print "The shape of the graph conv stages is as follows :"
test_feed = { graph.x: gardendata.test.images[:4], graph.y_: gardendata.test.labels[:4], graph.keep_prob:1.0, graph.training:False }
items = graph.conv
for item in graph.conv :
    print "    ",sess.run( item, test_feed ).shape,item.name

