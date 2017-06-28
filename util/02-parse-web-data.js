var URL_BASE = ['https://mygardenorg.s3.amazonaws.com/plantifier/' , 'https://s3-eu-west-1.amazonaws.com/mygardenorg/plantifier/']

var fs = require('fs');

var max_page = 1650

var current_page = 1

function parsePage(page) {
    var file_name = "../data/page"+page;
    console.log(file_name);
    var file = fs.readFileSync(file_name, "utf8");
    parts = file.split('<div id="plantifierGrid">')[1];
    parts = parts.split('</div>        <div class="clear">&nbsp;</div>')[0];
    parts = parts.split('<a id="photo');
    return parts;
}

function image_location(info) {
    return info.replace(/^.*src="(https:\/\/[^"]*)".*/,"$1").replace(/^.*plantifier\//,"").trim()
}

function confirmations(info) {
    if ( info.match(/identified as/) ) {
    	return 1
    }
    if ( info.replace(/^.*Current top answer .([0-9]+).*/,"$1") * 1 > 0 ) {
    	return 1
    }
    return 0
}

function get_label(info) {
    var result = info.replace(/^.*(urrent top answer |dentified as:).*<br\/><strong>/,"")
    result = result.replace(/<.*/,"")
    return result.replace(/^([0-9a-zA-Z ]+).*/,"$1").replace(/[^a-zA-Z ].*/,"").trim().toLowerCase()
}

var all_data = {}
for (var page = 1 ; page <= max_page ; page++) {
    data = parsePage(page)

    for (var index = 1; index < data.length ; index++) {
        all_data[ image_location(data[index]) ] = {
            'confirmed' : confirmations(data[index]),
            'label' : get_label(data[index])
        }
    }
}
console.log('saving file')
fs.writeFileSync( "../meta-data/all_data.json" , JSON.stringify(all_data) )