import numpy as np
import json

words = [ 'lily','rose','hydrangea','kalanchoe','hibiscus','begonia','hosta','iris','coleus','maple','geranium','peony','azalea','aloe','orchid','ivy','sedum','petunia','dracaena','dahlia','dianthus','red','virginia','bromeliad','japonica','magnolia','pothos','oak','schefflera','clematis','creeper','rhododendron','cyclamen','fern','jade','lantana','dieffenbachia','daisy','tulip','cactus','croton','columbine','celosia','ficus','hyacinth','yucca','salvia','canna','euphorbia','japanese','poppy','violet','palm','peace','poison','strawberry','day','camellia','morning','oxalis','impatiens','dogwood','phlox','glory','spathiphyllum','allium','vera','spider','vinca','african','wild','sweet','christmas','amaryllis','black' ]

confirmed = np.array(json.load( open("meta-data/confirmed_ids_with_top_words.json","r")))
all_data = json.load( open("meta-data/all_data.json","r") )


def labelToHot(label) :
    return [ word in label for word in words ]


labelMatrix = np.array([ labelToHot(all_data[imageName]['label']) for imageName in confirmed  ])


top_labels = ['hibiscus','begonia','hosta','iris','coleus','maple','geranium','peony','azalea','aloe','orchid','ivy','sedum','petunia','dracaena','dahlia','dianthus','bromeliad','japonica','magnolia','pothos','oak','schefflera','clematis','creeper','rhododendron','cyclamen','fern','jade','lantana','dieffenbachia','daisy','tulip','cactus','croton']
