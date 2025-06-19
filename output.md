Word Error Rate (WER) Results:
Whisper Large v3: 0.3452
Seamless M4T v2: 0.2110
Whisper Small: 0.6992

Number of valid documents after preprocessing: 198
PROCESSED: 40 ['hoe nals kooi. belangrik akademiese gedrag altyd eertis. heer zuid afrikaners verteenwoog hoop doorzettingsvermoe nasie ken maak. helias book. inzichtkomponente aangeduid aan einde voorgestelde antwoord elke vraag', 'skruewe gebruik slot aan deur vast te maak. hierdie wereld mens dinge vermaag waarvoor jy voor jyn net droom. evalueer media ander invloeder persoonlijke levensstylkeesers stel gepaslike reaksies voor. klippe pad gegooi verkeer te belemmer. mate ekonomische swaarkry ontevredenheid danige misdadige gedrag rechtvaardig', 'rekenplichtige beamte risikokampvechter voorzitter staatsbestuurskomitee. ak kyk soms talaviesie. parlements constitutionele hersieningskomitee verhoore alle provincies afgesluit. infrastruktuur stelsels loop gevaar verder achteruit te gaan uiteindelijk te misluk. lakker woonstel', 'aandag noodwendig deergans selfde persoon of persoone gevestig. hierdie gebouw verskaf inlichting oor mapungue bewaar kinsvoorwerpe antieke geschiedenis. ek gaan niewe skoene koop. afgekoelde oplossing concentratie bodieversadigingspit superversadig. elke school assesseringsprogram gegroond provinciale nationale richtlijnen ontwikkel', 'versoeken meer politie optrede gemeenskapsbetrokkenheid geimplementeer. elke raadslid ook persoonlik verantwoordelik gereelde betaling soedanige kontrak. mens vriendelik. door mekaar naag opstand verontrechting moedeloosheid selfbejammering. ek huis gaan']
PROCESSED: 40 ['neck cage. important academic conduct always honorable. lord south africans represents hope perseverance make nation known. helias book. insight components indicated end proposed answer question', 'used screws fasten lock door. world one could entertain things one could dream. evaluate media influencers personal lifestyle choices suggest appropriate responses. threw stones road obstruct traffic. measure economic hardship dissatisfaction justify petty criminal conduct', 'accountable officials risk fighter chairman state management committee. watch television sometimes. parliamentary constitutional review committee closed hearings provinces. infrastructure systems run risk falling behind ultimately failing. one much lighter apartment', 'attention necessarily focused person persons. building provides information mapungue preserves artifacts ancient history. go buy new shoes. cooled solution one concentration bodily saturation tip supersaturated. school must develop one assessment programme based provincial national guidelines', 'calls police action community involvement implemented. member council therefore personally responsible regular payment contract. man friendly. result mutual strife rebellion indignation one despondency self pity. want go home']
REF: 40 ['hou naels kou. belangrik akademiese gedrag altyd eties. hierdie suid afrikaners verteenwoordig hoop deursettingsvermo nasie kenmerk. lees boek. insigkomponente aangedui aan einde voorgestelde antwoord elke vraag', 'skroewe gebruik slot aan deur vas te maak. hierdie reld mens dinge vermag waaroor jy voorheen net droom. evalueer media ander invloede persoonlike lewenstylkeuses stel gepaste reaksies voor. klippe pad gegooi verkeer te belemmer. mate ekonomiese swaarkry ontevredenheid sodanige misdadige gedrag regverdig', 'rekenpligtige beampte risiko kampvegter voorsitter staatsbestuurskomitee. ek kyk soms televisie. parlement konstitusionele hersieningskomitee verhore alle provinsies afgesluit. infrastruktuurstelsels loop gevaar verder agteruit te gaan uiteindelik te misluk. lekker woonstel', 'aandag noodwendig deurgaans dieselfde persoon of persone gevestig. hierdie gebou verskaf inligting oor mapungubwe bewaar kunsvoorwerpe antieke geskiedenis. ek gaan nuwe skoene koop. afgekoelde oplossing konsentrasie bo versadigingspunt superversadig. elke skool assesseringsprogram gegrond provinsiale nasionale riglyne ontwikkel', 'versoeke meer polisieoptrede gemeenskapsbetrokkenheid ge mplementeer. elke raadslid ook persoonlik verantwoordelik gereelde betaling sodanige kontrak. mense vriendelik. deurmekaar nag opstand verontregting moedeloosheid selfbejammering. ek huis gaan']
REF: 40 ['stop chilling nails. important academic conduct always ethical. south africans represent hope resilience characterize nation. reading book. insight components indicated end proposed answer question', 'used screws fasten lock door. world could accomplish things could dream. evaluate media influences personal lifestyle choices suggest appropriate responses. threw stones road obstruct traffic. measure economic hardship dissatisfaction justify criminal conduct', 'accountable officer risk fighter chairman state management committee. watch television sometimes. parliament constitutional review committee closed hearings provinces. infrastructure systems run risk falling behind ultimately failing. nice apartment', 'attention necessarily focused person persons. building provides information mapungubwe preserves artifacts ancient history. go buy new shoes. cooled solution concentration saturation point supersaturated. school develop assessment program based provincial national guidelines', 'calls police action community involvement implemented. council member therefore personally responsible regular payment contract. people friendly. turbulent night rebellion defilement despair self pity. want go home']

# Whisper Large

Optimal number of topics: 2 with UMass: -11.17473779315901, NPMI: -0.45301230033027007
Optimal number of topics: 2 with UMass: -9.492261156187496, NPMI: -0.4188529394049015
Optimal number of topics: 6 with UMass: -14.036563960686651, NPMI: -0.45888852617321324
Optimal number of topics: 17 with UMass: -15.941290650906925, NPMI: -0.491132112239519

Optimal topics for Afrikaans (whisper_large): 2
Optimal topics for Afrikaans (Reference): 2
Optimal topics for English (whisper_large): 6
Optimal topics for English (Reference): 17

== Afrikaans ==

LDA (main):
LDA Topic 1: te, gebruik, aan, ook, gaan, ek, dier, verskillende, jy, maak
LDA Topic 2: ek, te, of, hierdie, jy, aan, maak, dienste, meer, hier

LDA (reference):
LDA Topic 1: jy, hierdie, ek, te, of, deur, aan, verduidelik, inligting, tot
LDA Topic 2: te, ek, hierdie, verskillende, gebruik, aan, deur, maak, leerders, gaan

BERTopic (main):
BERTopic 0: te, ek, hierdie, aan, verskillende, meer, ook, maak, gebruik, jy
BERTopic 1: te, aan, gebruik, dier, of, maak, verskaaf, door, alle, ook

BERTopic (reference):
BERTopic 0: media, ongelyke, te, mediese, gepaste, sodanige, lekker, samelewings, ekonomiese, ek
BERTopic 1: tot, of, jy, reeks, lank, siektes, veroorsaak, toegang, dienste, inligting

== English ==

LDA (main):
LDA Topic 1: one, time, south, differences, important, things, answer, result, services, components
LDA Topic 2: different, provided, one, access, information, human, police, animal, also, time
LDA Topic 3: one, process, per, go, explain, important, sports, animal, service, consider
LDA Topic 4: one, school, make, work, performed, new, international, develop, get, aware
LDA Topic 5: one, services, also, animal, use, culture, must, care, supports, employment
LDA Topic 6: one, risk, different, could, things, assessment, used, emergency, world, argument

LDA (reference):
LDA Topic 1: one, provided, also, old, consider, last, different, improved, companies, long
LDA Topic 2: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 3: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 4: process, products, sports, ongoing, unit, empty, long, companies, business, improved
LDA Topic 5: art, diseases, services, service, week, explain, health, community, academic, supports
LDA Topic 6: make, media, differences, night, achieving, economic, someone, nice, department, element
LDA Topic 7: cases, risk, assessment, says, things, wife, content, world, demand, population
LDA Topic 8: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 9: skills, go, together, different, learners, play, needed, work, aware, aspects
LDA Topic 10: form, help, time, using, areas, components, appointment, apartment, dinner, due
LDA Topic 11: school, get, go, assessment, new, person, concentration, provincial, international, national
LDA Topic 12: information, go, access, services, production, back, devices, answer, south, enough
LDA Topic 13: important, services, different, explain, municipal, farmers, takes, international, support, programme
LDA Topic 14: used, could, involves, rural, people, conduct, economic, eat, however, shapes
LDA Topic 15: new, things, also, use, please, arise, countries, currently, used, focus
LDA Topic 16: risk, services, students, children, provide, infrastructure, management, care, facilities, describe
LDA Topic 17: different, police, never, involvement, sustainable, principles, growth, always, service, projects

BERTopic (main):
BERTopic 0: one, school, go, develop, aware, unequal, provided, based, assessment, services
BERTopic 1: different, back, information, time, one, demand, population, production, never, range

BERTopic (reference):
BERTopic 0: risk, different, unequal, service, cases, important, information, services, systems, takes
BERTopic 1: provided, school, different, using, aware, one, assessment, also, play, needed

LDA Coherence scores for Afrikaans (whisper_large): {'umass': -11.17473779315901, 'npmi': -0.45301230033027007}
LDA Coherence scores for Afrikaans (Reference): {'umass': -9.492261156187496, 'npmi': -0.4188529394049015}
BERTopic Coherence scores for Afrikaans (whisper_large): {'umass': -6.11453595197484, 'npmi': -0.28789923494180897}
BERTopic Coherence scores for Afrikaans (Reference): {'umass': -10.741343423289173, 'npmi': -0.31318843572957833}
LDA Coherence scores for English (whisper_large): {'umass': -14.036563960686651, 'npmi': -0.45888852617321324}
LDA Coherence scores for English (Reference): {'umass': -15.941290650906925, 'npmi': -0.491132112239519}
BERTopic Coherence scores for English (whisper_large): {'umass': -10.40545866087843, 'npmi': -0.3678516076994684}
BERTopic Coherence scores for English (Reference): {'umass': -15.279125911313296, 'npmi': -0.5393976727471601}

# M4T v2

Number of valid documents after preprocessing: 198
PROCESSED: 40 ['nou goud. belangrik akademiese gedrag altyd eietis. hierdie suid afrikaners verteenwoordig hoop deursettingsvermo nasie vorm. hailius boek. insigkomponent aangedui aan einde voorgestelde antwoord elke vraag', 'skroewe gebruik slot aan deur vas te maak. hierdie reld mens dinge vermag waarvoor jy voorheen net droom. evalueer media ander invloede persoonlike leefstylkeuses stel gepaste reaksies voor. klippe pad gegooi verkeer te belemmer. mate ekonomiese swaarheid ontevredenheid sulke misdadige gedrag regverdig', 'rekenskaplike beampte risiko kampvegter voorsitter staatsbestuurskomitee. akkeeksom televisie. parlement konstitusionele hersieningskomitee verhoore alle provinsies afgesluit. infrastruktuurstelsels loop gevaar verder agteruit te gaan uiteindelik te misluk. lekker woonstel', 'aandag noodwendig deurgaans dieselfde persoon of persone gevestig. hierdie gebou verskaf inligting oor mapungubwe bewaar kunsvoorwerpe antieke geskiedenis. ek sak koop. afgekoelde oplossing konsentrasie bo versadigingspit superversadig. elke skool assesseringsprogram gegrond provinsiale nasionale riglyne ontwikkel', 'versoeke meer polisieoptrede gemeenskapsbetrokkenheid ge mplementeer. elke raadslid ook persoonlik verantwoordelik gereelde betaling sodanige kontrak. meeste mense vriendelik. deurmekaar nag opstand verontwaardiging moedeloosheid selfbejammering. ek haastig']
PROCESSED: 40 ['gold. therefore important academic conduct always ethical. south africans represent hope resilience make nation. hailius book. insight component indicated end proposed answer question', 'used screws fasten lock door. world could accomplish things could dream. evaluate media influences personal lifestyle choices propose appropriate responses. threw stones road obstruct traffic. measure economic hardship dissatisfaction justify criminal conduct', 'accounting officer risk campaigner chairman state management committee. akkeeksom television. parliament constitutional review committee closed hearings provinces. infrastructure systems run risk falling behind ultimately failing. nice apartment', 'attention necessarily focused person persons time. building provides information mapungubwe preserves artifacts ancient history. buy bag. cooled solution concentration saturation point supersaturated. school must develop assessment program based provincial national guidelines', 'calls police action community involvement implemented. council member therefore personally responsible regular payment contract. people friendly. turbulent night rebellion indignation despair self pity. hurry']
REF: 40 ['hou naels kou. belangrik akademiese gedrag altyd eties. hierdie suid afrikaners verteenwoordig hoop deursettingsvermo nasie kenmerk. lees boek. insigkomponente aangedui aan einde voorgestelde antwoord elke vraag', 'skroewe gebruik slot aan deur vas te maak. hierdie reld mens dinge vermag waaroor jy voorheen net droom. evalueer media ander invloede persoonlike lewenstylkeuses stel gepaste reaksies voor. klippe pad gegooi verkeer te belemmer. mate ekonomiese swaarkry ontevredenheid sodanige misdadige gedrag regverdig', 'rekenpligtige beampte risiko kampvegter voorsitter staatsbestuurskomitee. ek kyk soms televisie. parlement konstitusionele hersieningskomitee verhore alle provinsies afgesluit. infrastruktuurstelsels loop gevaar verder agteruit te gaan uiteindelik te misluk. lekker woonstel', 'aandag noodwendig deurgaans dieselfde persoon of persone gevestig. hierdie gebou verskaf inligting oor mapungubwe bewaar kunsvoorwerpe antieke geskiedenis. ek gaan nuwe skoene koop. afgekoelde oplossing konsentrasie bo versadigingspunt superversadig. elke skool assesseringsprogram gegrond provinsiale nasionale riglyne ontwikkel', 'versoeke meer polisieoptrede gemeenskapsbetrokkenheid ge mplementeer. elke raadslid ook persoonlik verantwoordelik gereelde betaling sodanige kontrak. mense vriendelik. deurmekaar nag opstand verontregting moedeloosheid selfbejammering. ek huis gaan']
REF: 40 ['stop chilling nails. important academic conduct always ethical. south africans represent hope resilience characterize nation. reading book. insight components indicated end proposed answer question', 'used screws fasten lock door. world could accomplish things could dream. evaluate media influences personal lifestyle choices suggest appropriate responses. threw stones road obstruct traffic. measure economic hardship dissatisfaction justify criminal conduct', 'accountable officer risk fighter chairman state management committee. watch television sometimes. parliament constitutional review committee closed hearings provinces. infrastructure systems run risk falling behind ultimately failing. nice apartment', 'attention necessarily focused person persons. building provides information mapungubwe preserves artifacts ancient history. go buy new shoes. cooled solution concentration saturation point supersaturated. school develop assessment program based provincial national guidelines', 'calls police action community involvement implemented. council member therefore personally responsible regular payment contract. people friendly. turbulent night rebellion defilement despair self pity. want go home']

Optimal number of topics: 2 with UMass: -8.385145420662493, NPMI: -0.36736249555810363
Optimal number of topics: 2 with UMass: -9.492261156187496, NPMI: -0.4188529394049015
Optimal number of topics: 12 with UMass: -15.570589418401154, NPMI: -0.4694712801781909
Optimal number of topics: 17 with UMass: -15.941290650906925, NPMI: -0.491132112239519

Optimal topics for Afrikaans (m4tv2): 2
Optimal topics for Afrikaans (Reference): 2
Optimal topics for English (m4tv2): 12
Optimal topics for English (Reference): 17

== Afrikaans ==

LDA (main):
LDA Topic 1: te, hierdie, ek, aan, gebruik, deur, nou, jy, maak, dinge
LDA Topic 2: te, ek, verskillende, hierdie, deur, aan, of, tot, gebruik, dienste

LDA (reference):
LDA Topic 1: jy, hierdie, ek, te, of, deur, aan, verduidelik, inligting, tot
LDA Topic 2: te, ek, hierdie, verskillende, gebruik, aan, deur, maak, leerders, gaan

BERTopic (main):
BERTopic 0: te, verskillende, tot, aan, ek, ongelyke, media, deur, tyd, ontstaan
BERTopic 1: gebruik, skool, te, verskaf, deur, weens, vroeg, deurgaans, oorgedra, aan

BERTopic (reference):
BERTopic 0: deur, te, hierdie, ek, aan, verskillende, jy, tot, elke, ook
BERTopic 1: te, gebruik, maak, jy, hoe, ek, verduidelik, aan, deur, oorgedra

== English ==

LDA (main):
LDA Topic 1: services, information, police, devices, due, associated, number, users, water, community
LDA Topic 2: academic, world, work, aware, association, associated, aspects, arts, art, argument
LDA Topic 3: different, important, therefore, officer, explain, transport, part, however, farmers, academic
LDA Topic 4: risk, also, media, culture, used, understand, element, resources, word, reads
LDA Topic 5: important, time, areas, make, human, service, sustainable, based, must, projects
LDA Topic 6: process, explain, service, please, cases, business, reporting, source, department, period
LDA Topic 7: last, could, school, make, world, today, night, eat, music, international
LDA Topic 8: form, process, different, apartment, develop, skills, must, learners, choices, language
LDA Topic 9: one, provided, different, management, access, rural, local, also, association, unit
LDA Topic 10: services, improved, infrastructure, health, municipal, shapes, old, enough, consider, back
LDA Topic 11: person, new, health, information, international, focused, time, develop, long, program
LDA Topic 12: use, art, also, must, promote, supports, night, currently, rural, focus

LDA (reference):
LDA Topic 1: one, provided, also, old, consider, last, different, improved, companies, long
LDA Topic 2: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 3: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 4: process, products, sports, ongoing, unit, empty, long, companies, business, improved
LDA Topic 5: art, diseases, services, service, week, explain, health, community, academic, supports
LDA Topic 6: make, media, differences, night, achieving, economic, someone, nice, department, element
LDA Topic 7: cases, risk, assessment, says, things, wife, content, world, demand, population
LDA Topic 8: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 9: skills, go, together, different, learners, play, needed, work, aware, aspects
LDA Topic 10: form, help, time, using, areas, components, appointment, apartment, dinner, due
LDA Topic 11: school, get, go, assessment, new, person, concentration, provincial, international, national
LDA Topic 12: information, go, access, services, production, back, devices, answer, south, enough
LDA Topic 13: important, services, different, explain, municipal, farmers, takes, international, support, programme
LDA Topic 14: used, could, involves, rural, people, conduct, economic, eat, however, shapes
LDA Topic 15: new, things, also, use, please, arise, countries, currently, used, focus
LDA Topic 16: risk, services, students, children, provide, infrastructure, management, care, facilities, describe
LDA Topic 17: different, police, never, involvement, sustainable, principles, growth, always, service, projects

BERTopic (main):
BERTopic 0: must, different, provided, explain, process, learners, saw, medical, management, students
BERTopic 1: information, services, unequal, water, different, increase, diseases, production, old, consider

BERTopic (reference):
BERTopic 0: school, get, aware, assessment, different, early, daily, facilities, answer, describe
BERTopic 1: unequal, service, explain, source, committee, increase, societies, support, ongoing, providers
LDA Coherence scores for Afrikaans (m4tv2): {'umass': -8.385145420662493, 'npmi': -0.36736249555810363}
LDA Coherence scores for Afrikaans (Reference): {'umass': -9.492261156187496, 'npmi': -0.4188529394049015}
BERTopic Coherence scores for Afrikaans (m4tv2): {'umass': -10.487128883952781, 'npmi': -0.42176067954784957}
BERTopic Coherence scores for Afrikaans (Reference): {'umass': -4.8179997883460555, 'npmi': -0.28007734858327166}
LDA Coherence scores for English (m4tv2): {'umass': -15.570589418401154, 'npmi': -0.4694712801781909}
LDA Coherence scores for English (Reference): {'umass': -15.941290650906925, 'npmi': -0.491132112239519}
BERTopic Coherence scores for English (m4tv2): {'umass': -16.84097309397443, 'npmi': -0.5432070370227199}
BERTopic Coherence scores for English (Reference): {'umass': -17.296171881348023, 'npmi': -0.48141950586716553}

# Whisper Small

Number of valid documents after preprocessing: 198
PROCESSED: 40 ['go jo naar go. belangrijk jouw academieza gedraag altijd eerdus. heer ze afrikaans verteen volg woop geurzittingsformaal nase maak. heelia book. de zich componente angedij aan einde vorgestalde antwoord ver elke vraag', 'had je screwver gebruik je sloot aan dier vast te maak. hier wereld mens dingen vermag waarvoor jy voor jen net droen. evalieer media ander fluida persoonlijke levenstijlkeesers stijlgepastelke reacties voer. alla klippen pad gechooi verkeerte belemer. mate ekonomische swaardgrij ontevredenheid soeddanige misdaardige gedrag draagferdig', 'de regentplichtige beamde risicul kampvechter voorzitter staatsbesteerskomitee. kijk soms televisie. parlementse konste tysionale herzienerskomitee had verwoeren alle provincies afgeslijn. infrastructieer staalsels leurt gevoor verder achter te gaan uiteindelijk te misluk. hate the bio lackered wind style', 'laandag nien uitwender diergans ook diezelfde persoon of persoonen gefeester geweest. heer gebouw verskaaf inlichten hoeheren mapunguwe bewaar kunstvoerwerpen zijn antieke geschiedenis. ek halm koner koop. afgekoelde oplossing had de concentratie boor verzadigingsbed super verzadig. elke school de assesseringsprogram gegrond provinciale nationale reglijne ontwakko', 'versukken meer politie optreden gemenskaps betrokkenheid ge mplementeer wordt. elke raadslid aan ook persoonlijk verantwoordelijk verie gereelde betalen soudane gecontraak. gmies menser boy frindelk. door mekaar naag opstand verontrechtning moedeloesheid zelfbejammering. aakle aastu choon']
PROCESSED: 40 ['go yeah go. therefore important academic conduct always honorable. lord rings represented africa following whip flavoring formula make noses. heelia book. components indicated end question', 'would used screwver tie hole animal. world could things could dream. equalize media others fluida personal lifestyle choices style painted feed reactions. alla made rocks paved way obstruct traffic. degree economic swordsmanship discontent justify criminal conduct', 'regent charge risicul fighter chairman state supervision committee. watch television sometimes. parliamentary arts crafts review committee disbanded provinces disbanded. steel infrastructure advantage going back ultimately failing. hate bio lackered wind styles', 'monday wild goose also feasting person persons. lord provides building lights highlights mapunguwe preserves artifacts ancient history. buy strawberries. cooled solution concentration drilled saturation bed super saturated. school must awaken assessment programme based provincial national guidelines', 'need implement police action community involvement. member council also personally responsible regular payments made sudanese contractors. funny man boy friend milk. result mutual rebellion indignation one discouragement self pity. aakle aastu choon']
REF: 40 ['hou naels kou. belangrik akademiese gedrag altyd eties. hierdie suid afrikaners verteenwoordig hoop deursettingsvermo nasie kenmerk. lees boek. insigkomponente aangedui aan einde voorgestelde antwoord elke vraag', 'skroewe gebruik slot aan deur vas te maak. hierdie reld mens dinge vermag waaroor jy voorheen net droom. evalueer media ander invloede persoonlike lewenstylkeuses stel gepaste reaksies voor. klippe pad gegooi verkeer te belemmer. mate ekonomiese swaarkry ontevredenheid sodanige misdadige gedrag regverdig', 'rekenpligtige beampte risiko kampvegter voorsitter staatsbestuurskomitee. ek kyk soms televisie. parlement konstitusionele hersieningskomitee verhore alle provinsies afgesluit. infrastruktuurstelsels loop gevaar verder agteruit te gaan uiteindelik te misluk. lekker woonstel', 'aandag noodwendig deurgaans dieselfde persoon of persone gevestig. hierdie gebou verskaf inligting oor mapungubwe bewaar kunsvoorwerpe antieke geskiedenis. ek gaan nuwe skoene koop. afgekoelde oplossing konsentrasie bo versadigingspunt superversadig. elke skool assesseringsprogram gegrond provinsiale nasionale riglyne ontwikkel', 'versoeke meer polisieoptrede gemeenskapsbetrokkenheid ge mplementeer. elke raadslid ook persoonlik verantwoordelik gereelde betaling sodanige kontrak. mense vriendelik. deurmekaar nag opstand verontregting moedeloosheid selfbejammering. ek huis gaan']
REF: 40 ['stop chilling nails. important academic conduct always ethical. south africans represent hope resilience characterize nation. reading book. insight components indicated end proposed answer question', 'used screws fasten lock door. world could accomplish things could dream. evaluate media influences personal lifestyle choices suggest appropriate responses. threw stones road obstruct traffic. measure economic hardship dissatisfaction justify criminal conduct', 'accountable officer risk fighter chairman state management committee. watch television sometimes. parliament constitutional review committee closed hearings provinces. infrastructure systems run risk falling behind ultimately failing. nice apartment', 'attention necessarily focused person persons. building provides information mapungubwe preserves artifacts ancient history. go buy new shoes. cooled solution concentration saturation point supersaturated. school develop assessment program based provincial national guidelines', 'calls police action community involvement implemented. council member therefore personally responsible regular payment contract. people friendly. turbulent night rebellion defilement despair self pity. want go home']

Optimal number of topics: 2 with UMass: -10.257249340512553, NPMI: -0.43715824343524357
Optimal number of topics: 2 with UMass: -9.492261156187496, NPMI: -0.4188529394049015
Optimal number of topics: 8 with UMass: -15.895473078182626, NPMI: -0.49303216559793384
Optimal number of topics: 17 with UMass: -15.941290650906925, NPMI: -0.491132112239519

Optimal topics for Afrikaans (whisper_small): 2
Optimal topics for Afrikaans (Reference): 2
Optimal topics for English (whisper_small): 8
Optimal topics for English (Reference): 17

== Afrikaans ==

LDA (main):
LDA Topic 1: aan, tot, verschillende, jouw, hier, voort, toegang, ek, of, de
LDA Topic 2: de, te, aan, je, ek, gebruik, door, als, zal, maak

LDA (reference):
LDA Topic 1: jy, hierdie, ek, te, of, deur, aan, verduidelik, inligting, tot
LDA Topic 2: te, ek, hierdie, verskillende, gebruik, aan, deur, maak, leerders, gaan

BERTopic (main):
BERTopic 0: je, aan, de, nou, te, wordt, zal, of, heet, gaan
BERTopic 1: te, gebruik, hier, jouw, school, onder, de, maak, door, beskrijf

BERTopic (reference):
BERTopic 0: te, ek, jy, elke, hierdie, of, ongelyke, tyd, media, aan
BERTopic 1: te, gebruik, maak, deur, aan, weens, oorgedra, aandete, leerder, vroeg

== English ==

LDA (main):
LDA Topic 1: important, argument, going, different, time, per, explain, saving, also, free
LDA Topic 2: one, word, continue, media, de, production, farmers, element, far, already
LDA Topic 3: services, arise, school, learner, process, beer, infrastructure, make, use, provide
LDA Topic 4: must, used, water, could, way, choices, areas, rural, learner, eat
LDA Topic 5: increase, one, go, man, population, women, milk, make, world, risk
LDA Topic 6: must, based, someone, bed, weather, service, important, one, school, parties
LDA Topic 7: day, forms, going, services, school, even, away, different, access, academic
LDA Topic 8: far, different, use, also, business, called, goods, clearly, provincial, history

LDA (reference):
LDA Topic 1: one, provided, also, old, consider, last, different, improved, companies, long
LDA Topic 2: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 3: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 4: process, products, sports, ongoing, unit, empty, long, companies, business, improved
LDA Topic 5: art, diseases, services, service, week, explain, health, community, academic, supports
LDA Topic 6: make, media, differences, night, achieving, economic, someone, nice, department, element
LDA Topic 7: cases, risk, assessment, says, things, wife, content, world, demand, population
LDA Topic 8: academic, assessment, aspects, art, arise, argument, areas, appropriate, appointment, users
LDA Topic 9: skills, go, together, different, learners, play, needed, work, aware, aspects
LDA Topic 10: form, help, time, using, areas, components, appointment, apartment, dinner, due
LDA Topic 11: school, get, go, assessment, new, person, concentration, provincial, international, national
LDA Topic 12: information, go, access, services, production, back, devices, answer, south, enough
LDA Topic 13: important, services, different, explain, municipal, farmers, takes, international, support, programme
LDA Topic 14: used, could, involves, rural, people, conduct, economic, eat, however, shapes
LDA Topic 15: new, things, also, use, please, arise, countries, currently, used, focus
LDA Topic 16: risk, services, students, children, provide, infrastructure, management, care, facilities, describe
LDA Topic 17: different, police, never, involvement, sustainable, principles, growth, always, service, projects

BERTopic (main):
BERTopic 0: one, different, production, increase, unequal, media, risk, arise, new, time
BERTopic 1: must, school, art, go, day, far, make, learner, choir, smoke

BERTopic (reference):
BERTopic 0: daily, learner, students, learners, must, rural, aware, due, develop, assessment
BERTopic 1: unequal, increase, emergency, societies, ongoing, medical, support, provided, process, explain
LDA Coherence scores for Afrikaans (whisper_small): {'umass': -10.257249340512553, 'npmi': -0.43715824343524357}
LDA Coherence scores for Afrikaans (Reference): {'umass': -9.492261156187496, 'npmi': -0.4188529394049015}
BERTopic Coherence scores for Afrikaans (whisper_small): {'umass': -6.92265326962295, 'npmi': -0.28660053164318666}
BERTopic Coherence scores for Afrikaans (Reference): {'umass': -9.076920052916723, 'npmi': -0.36996977482834714}
LDA Coherence scores for English (whisper_small): {'umass': -15.895473078182626, 'npmi': -0.49303216559793384}
LDA Coherence scores for English (Reference): {'umass': -15.941290650906925, 'npmi': -0.491132112239519}
BERTopic Coherence scores for English (whisper_small): {'umass': -18.30749240032972, 'npmi': -0.5032714309951503}
BERTopic Coherence scores for English (Reference): {'umass': -12.933981502746477, 'npmi': -0.30179085676179956}

Final wer_npmi_data: [{'model': 'whisper_large', 'wer': 0.34524384355383875, 'npmi_af_lda': -0.45301230033027007, 'npmi_en_lda': -0.45888852617321324, 'npmi_af_bertopic': -0.28789923494180897, 'npmi_en_bertopic': -0.3678516076994684, 'npmi_af_lda_reference': -0.4188529394049015, 'npmi_en_lda_reference': -0.491132112239519, 'npmi_af_bertopic_reference': -0.31318843572957833, 'npmi_en_bertopic_reference': -0.5393976727471601}, {'model': 'm4tv2', 'wer': 0.21100917431192662, 'npmi_af_lda': -0.36736249555810363, 'npmi_en_lda': -0.4694712801781909, 'npmi_af_bertopic': -0.42176067954784957, 'npmi_en_bertopic': -0.5432070370227199, 'npmi_af_lda_reference': -0.4188529394049015, 'npmi_en_lda_reference': -0.491132112239519, 'npmi_af_bertopic_reference': -0.28007734858327166, 'npmi_en_bertopic_reference': -0.48141950586716553}, {'model': 'whisper_small', 'wer': 0.69917914051183, 'npmi_af_lda': -0.43715824343524357, 'npmi_en_lda': -0.49303216559793384, 'npmi_af_bertopic': -0.28660053164318666, 'npmi_en_bertopic': -0.5032714309951503, 'npmi_af_lda_reference': -0.4188529394049015, 'npmi_en_lda_reference': -0.491132112239519, 'npmi_af_bertopic_reference': -0.36996977482834714, 'npmi_en_bertopic_reference': -0.30179085676179956}]

{
  "asr_results": {
    "whisper_large": {
      "transcripts": {
        "processed": [
          " Hoe op jou nals kooi?",
          " Dit is dus belangrik dat jou akademiese gedrag altyd eertis is.",
          " Heer die Zuid-Afrikaners verteenwoog die hoop en doorzettingsvermoe wat ons nasie ken maak.",
          " Helias, a book.",
          " Die inzichtkomponente word aangeduid aan die einde van die voorgestelde antwoord vir elke vraag."
        ],
        "reference": [
          "Hou op jou naels kou!",
          "Dit is dus belangrik dat jou akademiese gedrag altyd eties is.",
          "Hierdie Suid-Afrikaners verteenwoordig die hoop en deursettingsvermoë wat ons nasie kenmerk.",
          "Hy lees ’n boek",
          "Die insigkomponente word aangedui aan die einde van die voorgestelde antwoord vir elke vraag."
        ]
      },
      "wer": 0.34524384355383875,
      "translations": {
        "processed": [
          "How on your neck cage?",
          "So it is important that your academic conduct is always honorable.",
          "Lord the South Africans represents the hope and perseverance that make our nation known.",
          "Helias, a book.",
          "The insight components are indicated at the end of the proposed answer to each question."
        ],
        "reference": [
          "Stop chilling on your nails!",
          "So it is important that your academic conduct is always ethical.",
          "These South Africans represent the hope and resilience that characterize our nation.",
          "He's reading a book.",
          "The insight components are indicated at the end of the proposed answer to each question."
        ]
      },
      "topic_metrics": {
        "afrikaans": {
          "lda": {
            "umass": -11.17473779315901,
            "npmi": -0.45301230033027007
          },
          "bertopic": {
            "umass": -6.11453595197484,
            "npmi": -0.28789923494180897
          },
          "lda_reference": {
            "umass": -9.492261156187496,
            "npmi": -0.4188529394049015
          },
          "bertopic_reference": {
            "umass": -10.741343423289173,
            "npmi": -0.31318843572957833
          }
        },
        "english": {
          "lda": {
            "umass": -14.036563960686651,
            "npmi": -0.45888852617321324
          },
          "bertopic": {
            "umass": -10.40545866087843,
            "npmi": -0.3678516076994684
          },
          "lda_reference": {
            "umass": -15.941290650906925,
            "npmi": -0.491132112239519
          },
          "bertopic_reference": {
            "umass": -15.279125911313296,
            "npmi": -0.5393976727471601
          }
        }
      }
    },
    "m4tv2": {
      "transcripts": {
        "processed": [
          "Op jou nou is goud",
          "Dit is dus belangrik dat jou akademiese gedrag altyd eietis is",
          "Hierdie Suid-Afrikaners verteenwoordig die hoop en deursettingsvermoë wat ons nasie vorm",
          "Hailius 'n boek.",
          "Die insigkomponent word aangedui aan die einde van die voorgestelde antwoord vir elke vraag."
        ],
        "reference": [
          "Hou op jou naels kou!",
          "Dit is dus belangrik dat jou akademiese gedrag altyd eties is.",
          "Hierdie Suid-Afrikaners verteenwoordig die hoop en deursettingsvermoë wat ons nasie kenmerk.",
          "Hy lees ’n boek",
          "Die insigkomponente word aangedui aan die einde van die voorgestelde antwoord vir elke vraag."
        ]
      },
      "wer": 0.21100917431192662,
      "translations": {
        "processed": [
          "On you now is gold",
          "It is therefore important that your academic conduct is always ethical.",
          "These South Africans represent the hope and resilience that make up our nation",
          "Hailius a book.",
          "The insight component is indicated at the end of the proposed answer to each question."
        ],
        "reference": [
          "Stop chilling on your nails!",
          "So it is important that your academic conduct is always ethical.",
          "These South Africans represent the hope and resilience that characterize our nation.",
          "He's reading a book.",
          "The insight components are indicated at the end of the proposed answer to each question."
        ]
      },
      "topic_metrics": {
        "afrikaans": {
          "lda": {
            "umass": -8.385145420662493,
            "npmi": -0.36736249555810363
          },
          "bertopic": {
            "umass": -10.487128883952781,
            "npmi": -0.42176067954784957
          },
          "lda_reference": {
            "umass": -9.492261156187496,
            "npmi": -0.4188529394049015
          },
          "bertopic_reference": {
            "umass": -4.8179997883460555,
            "npmi": -0.28007734858327166
          }
        },
        "english": {
          "lda": {
            "umass": -15.570589418401154,
            "npmi": -0.4694712801781909
          },
          "bertopic": {
            "umass": -16.84097309397443,
            "npmi": -0.5432070370227199
          },
          "lda_reference": {
            "umass": -15.941290650906925,
            "npmi": -0.491132112239519
          },
          "bertopic_reference": {
            "umass": -17.296171881348023,
            "npmi": -0.48141950586716553
          }
        }
      }
    },
    "whisper_small": {
      "transcripts": {
        "processed": [
          " Go op, jo, naar is go.",
          " Dit is dus belangrijk dat jouw academieza gedraag altijd eerdus is.",
          " Heer die ze het afrikaans verteen volg die woop en geurzittingsformaal wat ons nase kan maak.",
          " Heelia is a book.",
          " De in zich componente word angedij aan die einde van die vorgestalde antwoord ver elke vraag."
        ],
        "reference": [
          "Hou op jou naels kou!",
          "Dit is dus belangrik dat jou akademiese gedrag altyd eties is.",
          "Hierdie Suid-Afrikaners verteenwoordig die hoop en deursettingsvermoë wat ons nasie kenmerk.",
          "Hy lees ’n boek",
          "Die insigkomponente word aangedui aan die einde van die voorgestelde antwoord vir elke vraag."
        ]
      },
      "wer": 0.69917914051183,
      "translations": {
        "processed": [
          "Go on, yeah, to is go.",
          "It is therefore important that your academic conduct is always honorable.",
          "The Lord of the Rings has represented Africa following the whip and flavoring formula that can make our noses.",
          "Heelia is a book.",
          "The components themselves are indicated at the end of each question."
        ],
        "reference": [
          "Stop chilling on your nails!",
          "So it is important that your academic conduct is always ethical.",
          "These South Africans represent the hope and resilience that characterize our nation.",
          "He's reading a book.",
          "The insight components are indicated at the end of the proposed answer to each question."
        ]
      },
      "topic_metrics": {
        "afrikaans": {
          "lda": {
            "umass": -10.257249340512553,
            "npmi": -0.43715824343524357
          },
          "bertopic": {
            "umass": -6.92265326962295,
            "npmi": -0.28660053164318666
          },
          "lda_reference": {
            "umass": -9.492261156187496,
            "npmi": -0.4188529394049015
          },
          "bertopic_reference": {
            "umass": -9.076920052916723,
            "npmi": -0.36996977482834714
          }
        },
        "english": {
          "lda": {
            "umass": -15.895473078182626,
            "npmi": -0.49303216559793384
          },
          "bertopic": {
            "umass": -18.30749240032972,
            "npmi": -0.5032714309951503
          },
          "lda_reference": {
            "umass": -15.941290650906925,
            "npmi": -0.491132112239519
          },
          "bertopic_reference": {
            "umass": -12.933981502746477,
            "npmi": -0.30179085676179956
          }
        }
      }
    }
  },
  "translation_results": {},
  "topic_results": {},
  "evaluation_metrics": {}
}