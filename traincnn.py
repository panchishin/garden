import tensorflow as tf
import numpy as np
import gardendata

NUM_CLASSES = len(gardendata.labels)

def parameter_count( shape , name="") :
  print name," Parametes ",shape,", Count :",reduce(lambda x, y: x*y, shape )

def weight_variable(shape, name="Weight_Variable"):
  parameter_count(shape,name)
  return tf.Variable( tf.truncated_normal(shape, stddev=0.01) )

def bias_variable(shape):
  return tf.Variable( tf.constant(0.1, shape=shape) )

def max_pool(x,stride=2):
  return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x,stride=2):
  return tf.nn.avg_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

def conv( x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="conv" ):
  w = weight_variable( [width, width, layers_in, layers_out] , name=name) 
  b = bias_variable( [layers_out] ) 
  return tf.nn.conv2d( x, w, strides= [1, stride, stride, 1], padding=padding ) + b

def conv_relu( x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="conv_relu" ):
  h = conv( x , layers_in , layers_out , width , stride, padding, name=name )
  return tf.nn.relu( h )

def batch_normalization( x, training, momentum=0.9 ) :
  return tf.layers.batch_normalization( x, training=training, momentum=momentum )


def handler_wrapper( handler ) :
  tf.reset_default_graph()
  x              = tf.placeholder( tf.float32, [None, 120, 120, 3], name="image_in" )
  y_             = tf.placeholder( tf.float32, [None, NUM_CLASSES], name="label_in" )
  keep_prob      = tf.placeholder( tf.float32, name="keep_prob" )
  learning_rate  = tf.placeholder( tf.float32, name="learning_rate" )
  training       = tf.placeholder( tf.bool, name="is_training_cycle" )
  h_pool         = handler.convolve( x, training,  keep_prob )
  y              = tf.reshape(h_pool, [-1,NUM_CLASSES], name="label_out")
  loss           = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), name="loss")
  update_ops     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train        = tf.train.AdamOptimizer(learning_rate).minimize(loss, name="train")
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_prediction")
  percent_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="percent_correct")
  return x,y_,y,keep_prob,loss,train,percent_correct,training,learning_rate


class cnn_graph :
  def convolve( self, image, training,  keep_prob ) :
    result = image
    result = batch_normalization( result, training, momentum=0.9 ) # [ 120,120 ]
    result = conv_relu( result , 3 , 32 , width=11 , stride=5, padding="VALID" )
    result = conv_relu( result , 32 , 64, width=5 , padding="VALID")
    result = max_pool(result )
    result = conv_relu( result , 64 , 64, width=3 , padding="VALID")
    result = conv_relu( result , 64 , 64, width=3 , padding="VALID")
    result = tf.nn.dropout( result , keep_prob )
    result = conv( result , 64 , NUM_CLASSES, width=1 )
    result = avg_pool(result, 5 )
    return result



handler = handler_wrapper(cnn_graph())
x,y_,y,keep_prob,loss,train,percent_correct,training,learning_rate = handler

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
sess.run( tf.local_variables_initializer() )

test_feed = { x: gardendata.test.images, y_: gardendata.test.labels, keep_prob:1.0, training:False }

LEARNING_RATE = 1e-4
def epochCycle(epoch):
  gardendata.train.shuffle()
  batch_size = 200
  batches = gardendata.train.labels.shape[0] / batch_size
  for batch in range(batches) :
    start_index = batch * batch_size
    end_index = start_index + batch_size
    train_feed = { x: gardendata.train.images[start_index:end_index], 
                  y_: gardendata.train.labels[start_index:end_index], 
                  keep_prob:0.5, 
                  learning_rate:LEARNING_RATE, 
                  training:True }
    result_loss, result_correct , _ = sess.run([loss,percent_correct,train], feed_dict=train_feed)
    if batch == batches - 1 :
        print "epoch",epoch,"error",(1.0-result_correct),"loss",result_loss,
        result_loss, result_correct = sess.run([loss,percent_correct], feed_dict=test_feed)
        print "  test error",(1.0-result_correct),"loss",result_loss


for epoch in range(200):
    epochCycle(epoch)


def confusionMatrix(y_in,y_out) :
    confusion = np.zeros( [NUM_CLASSES,NUM_CLASSES] , dtype=int )
    for index in range(y_in.shape[0]) :
        confusion[ np.argmax(y_in,1)[index] , np.argmax(y_out,1)[index] ] += 1
    return confusion

y_in , y_out = sess.run([y_,y],test_feed)
print confusionMatrix(y_in,y_out)

def buildRank(y_in,y_out,depth=5) :
    ranks = np.argmax( ( np.argsort(y_out,1) == np.argmax(y_in,1).reshape([-1,1]))[:,-depth:] ,1)
    print [ (num,np.sum( ranks == num )) for num in range(depth) ]
    return ranks

ranks = buildRank(y_in,y_out)
