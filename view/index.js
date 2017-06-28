var garden = angular.module('garden' , []);


garden.controller('gardenController', function ($scope,$http,$timeout,$interval) {

    $scope.data = {
    	'labels' : [] ,
        'predictedlabels' : [] ,
        'similarImages' : [] ,
        'text' : "this is the text" ,
        'current' : { 'name' : '44358_l.jpg' , 'image' : 'https://mygardenorg.s3.amazonaws.com/plantifier/44358_l.jpg' }
    }

    $http({ method : "GET" , url : "/labels" , cache: false}).then(function successCallback(result) {
        $scope.data["text"] = result.data
        $scope.data.labels = result.data
    })

    $scope.randomizeImage = function() {
    	$scope.data.predictedlabels = []
    	$scope.data.similarImages = []
        $http({ method : "GET" , url : "/random" , cache: false}).then(function successCallback(result) {
            $scope.data.current = result.data
            $scope.data.current.text=""
            $scope.predictLabels()
            $scope.showSimilarImages()
        })
    }

    $scope.predictLabels = function() {
        $http({ method : "GET" , url : "/classify/" + $scope.data.current.name , cache: false}).then(function successCallback(result) {
            $scope.data.predictedlabels = result.data.prediction
            $scope.data.current.text = result.data['meta-data'].label
        })
    }

    $scope.showSimilarImages = function() {
        $http({ method : "GET" , url : "/similar/" + $scope.data.current.name , cache: false}).then(function successCallback(result) {
            $scope.data.similarImages = result.data
        })
    }

    $scope.focusOnThisImage = function(info) {
    	$scope.data.current=info
    	$scope.data.current.text=""
    	$scope.predictLabels()
    }

    $scope.predictLabels();
    $scope.showSimilarImages();


});