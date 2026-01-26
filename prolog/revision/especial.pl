:-dynamic by/3.
% by(Character, Movie, Actor)
by(jackRyan, theSumOfAllFears, benAffleck).
by(cathyMuller, theSumOfAllFears, bridgetMoynahan).
by(jackRyan, theHuntForRedOctober, alecBaldwin).
by(jackRyan, patriotGames, harrisonFord).
by(cathyMuller, patriotGames, anneArcher).
by(jackRyan, clearAndPresentDanger, harrisonFord).
by(cathyMuller, clearAndPresentDanger, anneArcher).
by(president, airForceOne, harrisonFord).
by(frasierCrane, cheers, kelseyGrammer).
by(frasierCrane, frasier, kelseyGrammer).
by(rachelGreen, friends, jenniferAniston).
by(monicaGeller, friends, courteneyCox).
by(phoebeBuffay, friends, lisaKudrow).
by(ursulaBuffay, friends, lisaKudrow).
by(joeyTribbiani, friends, mattLeBlanc).
by(joeyTribbiani, joey, mattLeBlanc).
by(alexGarrett, joey, andreaAnders).
by(stephenColbert, dailyShow, stephenColbert).
by(stephenColbert, theColbertReport, stephenColbert).
by(addisonMontgomery, privatePractice, kateWalsh).
by(addisonMontgomery, greysAnatomy, kateWalsh).
by(mattMurdock, daredevil, benAffleck).
by(elektraNatchios, daredevil, jenniferGarner).
by(elektraNatchios, elektra, jenniferGarner).
by(elektraNatchios, elektra, lauraWard).
by(sydneyBristow, alias, jenniferGarner).

%plays_twins(?Actor, ?Movie)
plays_twins(Actor, Movie) :-
    by(Character1, Movie, Actor),
    by(Character2, Movie, Actor),
    Character1 \= Character2.

    