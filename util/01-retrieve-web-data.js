var http = require('https');
var fs = require('fs');

var max_page = 1650

var current_page = 14

function getPage() {
	page = current_page;
	current_page++;
	var file_name = "../data/page"+page;
	console.log(file_name);
	var file = fs.createWriteStream(file_name);
	var request = http.get("https://www.mygarden.org/plants/plantifier/identified?page="+page, function(response) {
		response.pipe(file);
		current_page++
		if (current_page <= max_page) {
			setTimeout( getPage , 2000 );
		}
	});	
}

getPage();