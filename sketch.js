
        
        
        
        
        
        
        
// __   _
// _(  )_( )_
// (_   _    _)
// (_) (__)













//   .-~~~-.
// .- ~ ~-(       )_ _
// /                     ~ -.
// |                           \
// \                         .'
// ~- . _____________ . -~













//   .-~~~-.
// .- ~ ~-(       )_ _
// /                    ~ -.
// |                          ',
// \                         .'
// ~- ._ ,. ,.,.,., ,.. -~
//    '       '







   


//                 _                                  
//       (`  ).                   _           
//      (     ).              .:(`  )`.       
// )           _(       '`.          :(   .    )      
// .=(`(      .   )     .--  `.  (    ) )      
// ((    (..__.:'-'   .+(   )   ` _`  ) )                 
// `.     `(       ) )       (   .  )     (   )  ._   
// )      ` __.:'   )     (   (   ))     `-'.-(`  ) 
// )  )  ( )       --'       `- __.'         :(      )) 
// .-'  (_.'          .')                    `(    )  ))
//           (_  )                     ` __.:'          
                                
// --..,___.--,--'`,---..-.--+--.,,-,,..._.--..-._.-a:f--.












//       .
                           
//       |					
// .               /				
// \       I     				
//           /
// \  ,g88R_
//   d888(`  ).                   _
// -  --==  888(     ).=--           .+(`  )`.
// )         Y8P(       '`.          :(   .    )
// .+(`(      .   )     .--  `.  (    ) )
// ((    (..__.:'-'   .=(   )   ` _`  ) )
// `.     `(       ) )       (   .  )     (   )  ._
// )      ` __.:'   )     (   (   ))     `-'.:(`  )
// )  )  ( )       --'       `- __.'         :(      ))
// .-'  (_.'          .')                    `(    )  ))
//           (_  )                     ` __.:'
                                    
// --..,___.--,--'`,---..-.--+--.,,-,,..._.--..-._.-a:f--.














//                                 ___    ,'""""'.
//                             ,"""   """"'      `.
//                            ,'        _.         `._
//                           ,'       ,'              `"""'.
//                          ,'    .-""`.    ,-'            `.
//                         ,'    (        ,'                :
//                       ,'     ,'           __,            `.
//                 ,""""'     .' ;-.    ,  ,'  \             `"""".
//               ,'           `-(   `._(_,'     )_                `.
//              ,'         ,---. \ @ ;   \ @ _,'                   `.
//         ,-""'         ,'      ,--'-    `;'                       `.
//        ,'            ,'      (      `. ,'                          `.
//        ;            ,'        \    _,','                            `.
//       ,'            ;          `--'  ,'                              `.
//      ,'             ;          __    (                    ,           `.
//      ;              `____...  `78b   `.                  ,'           ,'
//      ;    ...----'''' )  _.-  .d8P    `.                ,'    ,'    ,'
// _....----''' '.        _..--"_.-:.-' .'        `.             ,''.   ,' `--'
//       `" mGk "" _.-'' .-'`-.:..___...--' `-._      ,-"'   `-'
// _.--'       _.-'    .'   .' .'               `"""""
// __.-''        _.-'     .-'   .'  /
// '          _.-' .-'  .-'        .'
// _.-'  .-'  .-' .'  .'   /
// _.-'      .-'   .-'  .'   .'
// _.-'       .-'    .'   .'    /
// _.-'    .-'   .'    .'
// .-'            .'








// Art by Michal 'Goldmoon' Kwasniewski



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
        "start": ["Quite the cloudy day, but what do the clouds say?         #story#"],
        "story" : ["#greet#, I am a #adj# #cloud#. Some say I look like a #ans#, and that's just #feels#. Sometimes it gets rather #noise# up here. I miss #miss#. The other day, I heard the #animal# say - #pun# Alright, #wish# kid."],
        "cloud": ['cirrocumulus', 'cumulonimbus', 'nimbostratus','cirrus'],
        "greet" : ["Hi","Howdy", "Hey hey hey","Sup",'Yooohoooo'],
        "remind": ["poop", "strawberry shortcake", "butter chicken", "camp"],
        "noise" :['lonely','quiet','crowded','noisy','KA-BOOM','SKADOOSH'],
        "feels" : ["outrageous", "alarming", "wonderful", "disgusting", "lovely",'skrtskrt'],
        "miss" : ["sun-bathing", "wet parties", "diving", "skiing"],
        "animal" : ["raccoon", "crow", "ufo","helicopter", "dodo",'gremlin','dragon','phoenix','Centaur','Unicorn','Pegasus'],
        "pun" : ["two’s company, three’s a cloud", "What sort of clothes do clouds have? Thunderwear.","Why do metrologists pay so much attention to wispy clouds? They take them cirrus-ly.", "I’d tell you a joke about a cloud but it would be over your head.", "What is a cloud’s favourite reptile? A blizzard.","What’s worse than rain clouds? When it’s hailing taxis."],
        "adj" : ['tremendous', 'gooey', 'sharp', 'dark', 'scary','flabbergasted','Scrumptious','Splendid', 'Smelly','lit'],
        "wish":["Scurry along", "Stay safe","Go catch another", "Live life",'Toodles'],
        "ans": ans
    }
    var grammar = tracery.createGrammar(story)
    var result = grammar.flatten("#start#")
    document.getElementById('story').style.display = "block"
    document.getElementById('desc').innerHTML = result

    console.log(result)
}


