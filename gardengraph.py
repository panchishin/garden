import tensorflow as tf
import layer

class cnn :
  def __init__( self, num_classes , image_size=128 , embedding_size=128 ) :
    self.SIZE = image_size
    self.EMBED_SIZE = embedding_size

    self.num_classes    = num_classes

    self.x              = tf.placeholder( tf.float32, [None, self.SIZE, self.SIZE, 3], name="image_in" )
    self.y_             = tf.placeholder( tf.float32, [None, num_classes], name="label_in" )
    self.keep_prob      = tf.placeholder( tf.float32, name="keep_prob" )
    self.learning_rate  = tf.placeholder( tf.float32, name="learning_rate" )
    self.training       = tf.placeholder( tf.bool, name="is_training_cycle" )
    
    stages = []

    stages.append( layer.batch_normalization( self.x , training=self.training ) )
    stages.append( layer.conv_relu( stages[-1] , 3 , 16 , width=7, padding='SAME', name="3x16" ) )

    stages.append( layer.resnet_downscale( stages[-1] ) )
    stages.append( layer.resnet_block( stages[-1] , 32, 3 , training=self.training, momentum=0.99, name="32" ) )

    stages.append( layer.resnet_downscale( stages[-1] ) )
    stages.append( layer.resnet_block( stages[-1] , 64, 3 , training=self.training, momentum=0.99, name="64" ) )

    stages.append( layer.resnet_downscale( stages[-1] ) )
    stages.append( layer.resnet_narrow( stages[-1] , 128, 3 , training=self.training, narrowing=4, momentum=0.99, name="128_1" ) )

    stages.append( layer.resnet_downscale( stages[-1] ) )
    stages.append( layer.resnet_narrow( stages[-1] , 256, 3 , training=self.training, narrowing=8, momentum=0.99, name="256_1" ) )

    stages.append( layer.resnet_downscale( stages[-1] ) )
    stages.append( layer.resnet_narrow( stages[-1] , 512, 3 , training=self.training, narrowing=16, momentum=0.99, name="512_1" ) )

    stages.append( layer.avg_pool( stages[-1] , stride=4 ) )
    stages.append( layer.conv_relu( stages[-1] , 512 , self.EMBED_SIZE , width=1, padding='SAME', name="512xEmbed" ) )
    stages.append( tf.nn.sigmoid( tf.reshape(stages[-1], [-1,self.EMBED_SIZE] ), name="image_to_vector") )

    stages.append( tf.nn.dropout( stages[-1] , self.keep_prob , name="dropout" )  )
    stages.append( layer.fully_connected( stages[-1], self.EMBED_SIZE, num_classes, name="label_out") )

    self.conv = stages

    self.y              = self.conv[-1]
    self.y_softmax      = tf.nn.softmax(self.y)

    self.loss           = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_), name="loss")
    self.update_ops     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(self.update_ops):
      self.train        = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="train")

    self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1), name="correct_prediction")
    self.percent_correct = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="percent_correct")


