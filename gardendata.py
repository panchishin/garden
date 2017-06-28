import topwords
import images
import numpy as np

labels = ['hibiscus','begonia','hosta','iris','coleus','maple','geranium','peony','azalea','aloe','orchid','ivy','sedum','petunia','dracaena','dahlia','dianthus','bromeliad','japonica','magnolia','pothos','oak','schefflera','clematis','creeper','rhododendron','cyclamen','fern','jade','lantana','dieffenbachia','daisy','tulip','cactus','croton']
labels = ['cactus','tulip','peony','fern']

def get_files_for_label(label) :
    return topwords.confirmed[ topwords.labelMatrix[:,topwords.words.index(label)] == 1 ]

def create_image_data(label_index) :
    one_hot = np.zeros( [len(labels)] )
    one_hot[label_index] = 1
    return images.from_names().load( get_files_for_label( labels[label_index] ) , label=one_hot ) 

data_set = [ create_image_data(label_index) for label_index in range(len(labels)) ]

test_size = 20
test = data_set[0].slice(end=test_size)
train = data_set[0].slice(start=test_size)

for data in data_set[1:] :
    test.concat( data.slice(end=test_size) )
    train.concat( data.slice(start=test_size) )


