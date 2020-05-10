var loadFile = function (event) {
    var image = document.getElementById('output');
    image.src = URL.createObjectURL(event.target.files[0]);
    Model();
};

var ans = []

function Model() {
    ml5
        .imageClassifier("MobileNet")
        .then(classifier => classifier.classify(document.getElementById('output')))
        .then(results => {
            console.log(results)
            ans = []
            results.forEach(data => ans.push(data.label))
            document.getElementById('results').innerHTML = "This looks a bit like " + ans
            console.log(ans)
            GenerateStory();

        });
}

function GenerateStory() {
    var story = {
        "start": ["Quite the cloudy day, but what do the clouds say? A #ans# is flying towards you"],
        "ans": ans
    }
    var grammar = tracery.createGrammar(story)
    var result = grammar.flatten("#start#")
    document.getElementById('story').style.display = "block"
    document.getElementById('desc').innerHTML = result

    console.log(result)
}


