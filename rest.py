import falcon
import json
import topwords
from scipy import spatial

labels = topwords.top_labels

class Ping:
    def on_get(self, req, resp):
        resp.body = json.dumps( { 'response': 'ping' } )

class Labels:
    def on_get(self, req, resp):
        resp.body = json.dumps(labels)

class Classify:
    def on_get(self, req, resp, img):
    	#img = req.get_param('img')
        resp.body = json.dumps( topwords.all_data[img] )

class Similar:
    def on_get(self, req, resp, img):
    	# img = req.get_param('img')
    	distance = spatial.distance.cosine( [1.2,3.2] , [3.2,3.4] )
    	resp.body = json.dumps( [{ "img" : "100216_l.jpg" , "distance" : distance}] )


api = falcon.API()
api.add_route('/ping', Ping())
api.add_route('/labels', Labels())
api.add_route('/classify/{img}', Classify())
api.add_route('/similar/{img}', Similar())
