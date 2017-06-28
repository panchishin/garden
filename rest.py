import thread
import falcon
import json
import topwords
import numpy as np
import images
import random
import os.path
import tensorflow as tf
from scipy import spatial

labels = np.array(topwords.top_labels)


print """
================================
Restore the saved model
================================
"""
sess = tf.Session()
import gardengraph
graph = gardengraph.cnn(len(labels))
saver = tf.train.Saver()
saver.restore(sess,"meta-data/model")


print """
================================
Reader of images from disk
================================
"""
import images
img_reader = images.image_reader_graph()

def getOrDownloadImage(name) :
    if not os.path.isfile("data/"+name) :
        print "Don't have this one, downloading ..."
        images.download(name)
        print "... done"
    return getImage(name)

def getImage(name) :
    return sess.run(img_reader.img, {img_reader.img_name : "data/"+name } ).reshape([1,120,120,3])


print """
================================
Predict label definition
================================
"""
def predictLabel(image) :
    test_feed = { graph.x: image, graph.keep_prob:1.0, graph.training:False }
    y_out = sess.run(graph.y,test_feed)
    return labels[ np.argsort( y_out, 1 )[0,-5:] ][::-1]


print """
================================
Image to vector
================================
"""
vector = tf.get_default_graph().get_tensor_by_name("image_to_vector:0")

def imageToVector(image) :
    test_feed = { graph.x: image, graph.keep_prob:1.0, graph.training:False }
    return sess.run(vector,test_feed)[0,:]


name_vector_names = []
name_vector_vector =[]
def getTheVectors() :
  for label in topwords.top_labels :
    for name in topwords.get_files_for_label(label) :
        try :
            name_vector_vector.append( imageToVector( getImage( name ) ) )
            name_vector_names.append( name )
        except :
            pass

thread.start_new_thread( getTheVectors , () )


def distance(vector1,vector2) :
    return spatial.distance.cosine( vector1, vector2 )

def nearestNeighbour(image_vector) :
    distances = np.array([ distance(image_vector,reference) for reference in name_vector_vector ])
    nearest = np.argsort( distances )[:10]
    return np.array(name_vector_names)[nearest]



print """
================================
Define the rest endpoints
================================
"""

class Ping:
    def on_get(self, req, resp):
        resp.body = json.dumps( { 'response': 'ping' } )

class Labels:
    def on_get(self, req, resp):
        resp.body = json.dumps(labels.tolist())

class Classify:
    def on_get(self, req, resp, img):
        resp.body = json.dumps( {
            "name" : img,
            "image" : images.nameToURL(img),
            "meta-data" : topwords.all_data[img] ,
            "prediction" : predictLabel( getOrDownloadImage(img) ).tolist()
            } )

class Similar:
    def on_get(self, req, resp, img):
        names = nearestNeighbour(imageToVector(getOrDownloadImage(img))).tolist()
        result = [ {"name":name, "image":images.nameToURL(name)} for name in names ]
        resp.body = json.dumps( result )

class Display:
    def on_get(self, req, resp, file_name):
        if not os.path.isfile("view/"+file_name) :
            return

        result = open("view/"+file_name,"r")
        if ( "html" in file_name) :
            resp.content_type = "text/html"
        else :
            resp.content_type = "text/plain"
        
        resp.body = result.read()
        result.close()

class RandomName:
    def on_get(self, req, resp):
        name = random.choice(name_vector_names)
        resp.body = json.dumps( {"name":name, "image":images.nameToURL(name) } )


print """
================================
Add the endpoints to the service
================================
"""
api = falcon.API()
api.add_route('/ping', Ping())
api.add_route('/labels', Labels())
api.add_route('/classify/{img}', Classify())
api.add_route('/similar/{img}', Similar())
api.add_route('/view/{file_name}', Display())
api.add_route('/random', RandomName())



