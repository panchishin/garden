import tensorflow as tf

def parameter_count( shape , name="") :
  print name," Parametes ",shape,", Count :",reduce(lambda x, y: x*y, shape )

def weight_variable(shape, name="Weight_Variable"):
  parameter_count(shape,name)
  return tf.Variable( tf.truncated_normal(shape, stddev=0.01) )

def bias_variable(shape):
  return tf.Variable( tf.constant(0.1, shape=shape) )

def max_pool(x,stride=2, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=padding)

def avg_pool(x,stride=2, padding='SAME'):
  return tf.nn.avg_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=padding)

def conv( x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="conv" ):
  w = weight_variable( [width, width, layers_in, layers_out]) 
  b = bias_variable( [layers_out] ) 
  return tf.add( tf.nn.conv2d( x, w, strides= [1, stride, stride, 1], padding=padding ) , b , name=name)

def conv_relu( x , layers_in , layers_out , width=6 , stride=1, padding='SAME', name="conv_relu" ):
  h = conv( x , layers_in , layers_out , width , stride, padding )
  return tf.nn.relu( h , name=name)

def batch_normalization( x, training, momentum=0.9 ) :
  return tf.layers.batch_normalization( x, training=training, momentum=momentum )

def fully_connected( x , size_in , size_out, name="fully_connected" ):
  W = weight_variable( [size_in, size_out] )
  b = bias_variable( [size_out] )
  return tf.add( tf.matmul(x, W) , b , name=name )


class cnn :
  def __init__( self, num_classes ) :

    self.num_classes    = num_classes

    self.x              = tf.placeholder( tf.float32, [None, 120, 120, 3], name="image_in" )
    self.y_             = tf.placeholder( tf.float32, [None, num_classes], name="label_in" )
    self.keep_prob      = tf.placeholder( tf.float32, name="keep_prob" )
    self.learning_rate  = tf.placeholder( tf.float32, name="learning_rate" )
    self.training       = tf.placeholder( tf.bool, name="is_training_cycle" )

    self.conv = []
    
    self.conv.append( batch_normalization( self.x, self.training, momentum=0.9 )        )
    self.conv.append( conv_relu( self.conv[-1] , 3 , 32 , width=11 , stride=5, padding="VALID" ) )
    self.conv.append( conv_relu( self.conv[-1] , 32 , 64, width=5 , padding="VALID")    )
    self.conv.append( max_pool( self.conv[-1] )                                         )
    self.conv.append( conv_relu( self.conv[-1] , 64 , 64, width=3 , padding="VALID")    )
    self.conv.append( tf.nn.max_pool(self.conv[-1], ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID') )
    self.conv.append( tf.nn.max_pool(self.conv[-1], ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='VALID') )
    self.conv.append( tf.nn.sigmoid( tf.reshape(self.conv[-1], [-1,576] ), name="image_to_vector") )
    self.conv.append( tf.nn.dropout( self.conv[-1] , self.keep_prob , name="dropout" )  )
    self.conv.append( fully_connected( self.conv[-1], 576, num_classes, name="label_out") )

    self.y              = self.conv[-1]

    self.loss           = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_), name="loss")
    self.update_ops     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(self.update_ops):
      self.train        = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="train")

    self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1), name="correct_prediction")
    self.percent_correct = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="percent_correct")


