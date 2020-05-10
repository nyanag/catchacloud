
        
        
        
        
        
        
        
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
        "start": ["Quite the cloudy day, but what do the clouds say? #story#"],
        "story" : ["Woah! That's a #adj# #ans#. Reminds me of #remind#"],
        "remind": ["poop", "strawberry shortcake", "butter chicken", "camp"],
        "adj" : ['tremendous', 'gooey', 'sharp', 'dark', 'scary'],
        "ans": ans
    }
    var grammar = tracery.createGrammar(story)
    var result = grammar.flatten("#start#")
    document.getElementById('story').style.display = "block"
    document.getElementById('desc').innerHTML = result

    console.log(result)
}


