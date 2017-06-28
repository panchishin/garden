var URL_BASE = ['https://mygardenorg.s3.amazonaws.com/plantifier/' , 'https://s3-eu-west-1.amazonaws.com/mygardenorg/plantifier/']

var fs = require('fs');
var file = fs.readFileSync("../meta-data/all_data.json");
var data = JSON.parse(file)

var keys = Object.keys(data)
var labels = keys.filter( function(id) { return data[id].confirmed } ).map( function(id) { return data[id].label } )
var unique_labels = {}

var all_words = labels.join(" ").split(" ")
var unique_words = {}
for (var index in all_words) { unique_words[all_words[index]] = unique_words[all_words[index]] ? unique_words[all_words[index]]+1 : 1 }
console.log("Unique words : ",Object.keys(unique_words).length)
Object.keys(unique_words).filter( function(key) { return unique_words[key] > 100 } )

var top_words = [ 'lily','rose','hydrangea','kalanchoe','hibiscus','begonia','hosta','iris','coleus','maple','geranium','peony','azalea','aloe','orchid','ivy','sedum','petunia','dracaena','dahlia','dianthus','red','virginia','bromeliad','japonica','magnolia','pothos','oak','schefflera','clematis','creeper','rhododendron','cyclamen','fern','jade','lantana','dieffenbachia','daisy','tulip','cactus','croton','columbine','celosia','ficus','hyacinth','yucca','salvia','canna','euphorbia','japanese','poppy','violet','palm','peace','poison','strawberry','day','camellia','morning','oxalis','impatiens','dogwood','phlox','glory','spathiphyllum','allium','vera','spider','vinca','african','wild','sweet','christmas','amaryllis','black' ]
var top_word_lookup = {}
for (index in top_words) { top_word_lookup[ top_words[index] ] = 1 }

function hasTopWord(description) {
    words = description.split(" ");
    for (index in words) {
        if ( top_word_lookup[words[index]] ) {
            return true;
        }
    }
    return false;
}

var confirmed_ids_with_top_words = keys.filter( function(id) { return data[id].confirmed } ).filter( function(id) { return hasTopWord( data[id].label ) } )

fs.writeFileSync("../meta-data/confirmed_ids_with_top_words.json",JSON.stringify(confirmed_ids_with_top_words));

console.log("There are about 75 top words:\n")
console.log( top_words.join(",") )
console.log("\nand there are")
console.log(confirmed_ids_with_top_words.length )
console.log("images that are confirmed that have those words")
console.log("and")
console.log(keys.filter( function(id) { return !data[id].confirmed } ).filter( function(id) { return hasTopWord( data[id].label ) } ).length)
console.log("unconfirmed images that have those words")


