import tensorflow as tf
import numpy as np

IMAGE_SIZE = 128
print """

================================
Get the Garden Data
================================

"""
import gardendata
NUM_CLASSES = len(gardendata.labels)


print """

================================
Get the Garden Graph
================================

"""
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

def epochCycle(epoch,learning_rate=1e-4):
  gardendata.train.shuffle()

  def doWholeBatch( dataset , keep_prob=0.5, learning_rate=1e-4, training=True ) :
    batch_size = 50
    batches = dataset.labels.shape[0] / batch_size
    total_loss = 0.
    total_correct = 0.
    for batch in range(batches) :
      start_index = batch * batch_size
      end_index = start_index + batch_size
      train_feed = { graph.x: dataset.images[start_index:end_index], 
                  graph.y_: dataset.labels[start_index:end_index], 
                  graph.keep_prob:keep_prob, 
                  graph.learning_rate:learning_rate, 
                  graph.training:training }

      if training :
        result_loss, result_correct , _ = sess.run([graph.loss,graph.percent_correct,graph.train], feed_dict=train_feed)
      else :
        result_loss, result_correct = sess.run([graph.loss,graph.percent_correct], feed_dict=train_feed)

      total_loss += result_loss
      total_correct += result_correct

    total_loss = total_loss / batches
    total_correct = total_correct / batches

    return total_loss , total_correct

  result_loss , result_correct = doWholeBatch( gardendata.train , keep_prob=0.5, learning_rate=learning_rate, training=True )
  if epoch % 5 == 0 :
    print "epoch",epoch,"error",round(100*(1.0-result_correct),2),"%, loss",round(result_loss,4),
    result_loss , result_correct = doWholeBatch( gardendata.test , keep_prob=1.0, learning_rate=learning_rate, training=False )
    print "  test error",round(100*(1.0-result_correct),2),"%, loss",round(result_loss,4)


print """

================================
Do 250 training Epochs
================================

"""
for epoch in range(35):
    epochCycle(epoch,learning_rate=1e-5)

for epoch in range(35):
    epochCycle(epoch,learning_rate=3e-5)

for epoch in range(35):
    epochCycle(epoch,learning_rate=1e-5)




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
confuse = np.array(confusionMatrix(y_in,y_out))


print """

================================
How close were the estimates
================================

"""
def buildRank(y_in,y_out,depth=6) :
    ranks = np.argmax( ( np.argsort(y_out,1) == np.argmax(y_in,1).reshape([-1,1]))[:,-depth:] ,1)
    print np.sum( ranks > 0 ),"/",ranks.shape[0]," = %4.1f"%(100.0 * np.sum( ranks > 0 ) / ranks.shape[0]),"% are in the top",(depth-1)
    print "This is the breakdown (position 5 = top , position 0 = flop) :"
    for result in [ (num,np.sum( ranks == num )) for num in range(depth) ] :
       print "    position",result[0],"count",result[1]
    return ranks

ranks = buildRank(y_in,y_out)

labels = np.array(gardendata.labels)
accuracy = 100. * np.array([ confuse[i,i] for i in range(confuse.shape[0]) ]) / np.sum( confuse , 0 ) 
print "The most accurate labels are :",labels[ np.argsort( accuracy )[-1:-11:-1] ]


print """

================================
Save the model to disk
================================

"""
saver = tf.train.Saver()
saver.save(sess,"meta-data/model")
