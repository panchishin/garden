import tensorflow as tf
import numpy as np
import gardendata
NUM_CLASSES = len(gardendata.labels)


import gardengraph
graph = gardengraph.cnn(NUM_CLASSES)

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
sess.run( tf.local_variables_initializer() )


print """

================================
Show the graph sizes for
processing a batch of 4 images
================================

"""
test_feed = { graph.x: gardendata.test.images[:4], graph.y_: gardendata.test.labels[:4], graph.keep_prob:1.0, graph.training:False }
items = graph.conv
for item in graph.conv :
    print "    ",sess.run( item, test_feed ).shape,item.name


print """

================================
Define the training cycle
================================

"""
test_feed = { graph.x: gardendata.test.images, graph.y_: gardendata.test.labels, graph.keep_prob:1.0, graph.training:False }
LEARNING_RATE = 1e-4
def epochCycle(epoch):
  gardendata.train.shuffle()
  batch_size = 50
  batches = gardendata.train.labels.shape[0] / batch_size
  for batch in range(batches) :
    start_index = batch * batch_size
    end_index = start_index + batch_size
    train_feed = { graph.x: gardendata.train.images[start_index:end_index], 
                  graph.y_: gardendata.train.labels[start_index:end_index], 
                  graph.keep_prob:0.5, 
                  graph.learning_rate:LEARNING_RATE, 
                  graph.training:True }
    result_loss, result_correct , _ = sess.run([graph.loss,graph.percent_correct,graph.train], feed_dict=train_feed)
    if batch == batches - 1 and epoch % 5 == 0 :
        print "epoch",epoch,"error",(1.0-result_correct),"loss",result_loss,
        result_loss, result_correct = sess.run([graph.loss,graph.percent_correct], feed_dict=test_feed)
        print "  test error",(1.0-result_correct),"loss",result_loss


print """

================================
Do 500 training Epochs
================================

"""
for epoch in range(500):
    epochCycle(epoch)




print """

================================
Show the confusion matrix
================================

"""
def confusionMatrix(y_in,y_out) :
    confusion = np.zeros( [NUM_CLASSES,NUM_CLASSES] , dtype=int )
    for index in range(y_in.shape[0]) :
        confusion[ np.argmax(y_in,1)[index] , np.argmax(y_out,1)[index] ] += 1
    return confusion

y_in , y_out = sess.run([graph.y_,graph.y],test_feed)
print confusionMatrix(y_in,y_out)

print """

================================
How close were the estimates
================================

"""
def buildRank(y_in,y_out,depth=5) :
    ranks = np.argmax( ( np.argsort(y_out,1) == np.argmax(y_in,1).reshape([-1,1]))[:,-depth:] ,1)
    print [ (num,np.sum( ranks == num )) for num in range(depth) ]
    return ranks

ranks = buildRank(y_in,y_out,NUM_CLASSES)
