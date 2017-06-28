import tensorflow as tf

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


def cnn_core( image, training, keep_prob, num_classes ) :
    result = image
    result = batch_normalization( result, training, momentum=0.9 ) # [ 120,120 ]
    result = conv_relu( result , 3 , 32 , width=11 , stride=5, padding="VALID" )
    result = conv_relu( result , 32 , 64, width=5 , padding="VALID")
    result = max_pool(result )
    result = conv_relu( result , 64 , 64, width=3 , padding="VALID")
    result = conv_relu( result , 64 , 64, width=3 , padding="VALID")
    result = tf.nn.dropout( result , keep_prob )
    result = conv( result , 64 , num_classes, width=1 )
    result = avg_pool(result, 5 )
    return result


class cnn :
  def __init__( self, num_classes, handler = None ) :
    if handler == None :
      handler = cnn_core

    self.num_classes    = num_classes

    self.x              = tf.placeholder( tf.float32, [None, 120, 120, 3], name="image_in" )
    self.y_             = tf.placeholder( tf.float32, [None, num_classes], name="label_in" )
    self.keep_prob      = tf.placeholder( tf.float32, name="keep_prob" )
    self.learning_rate  = tf.placeholder( tf.float32, name="learning_rate" )
    self.training       = tf.placeholder( tf.bool, name="is_training_cycle" )

    self.h_pool         = handler( self.x, self.training, self.keep_prob, num_classes )
    self.y              = tf.reshape(self.h_pool, [-1,num_classes], name="label_out")

    self.loss           = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_), name="loss")
    self.update_ops     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(self.update_ops):
      self.train        = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="train")

    self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1), name="correct_prediction")
    self.percent_correct = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="percent_correct")


