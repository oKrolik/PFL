bird(robinho,    robin,   male,   [red, brown, white]).
bird(robina,     robin,   female, [brown, red, white]).
bird(ferrugem,   robin,   male,   [brown, gray]).

bird(arcoiris,   parrot,  male,   [red, blue, green, yellow]).
bird(verdeja,    parrot,  female, [green, yellow]).

bird(minerva,    owl,     female, [brown, white]).
bird(noctis,     owl,     male,   [gray, white]).
bird(sabia,      owl,     female, [beige, brown]).

male(B) :- bird(B, _, male, _).

has_more_color_of(Name, GreaterColor, LesserColor):-
    bird(Name, _Species, _Gender, Colors),
    append(_, [GreaterColor | Rest], Colors),
    append(_, [LesserColor | _], Rest).
    
most_colorful(Species, Name, NColors) :-
    bird(Name, Species, _Gender, Colors),
    length(Colors, NColors),
    \+ (
        bird(OtherName, Species, _OtherGender, OtherColors),
        OtherName \= Name,
        length(OtherColors, OtherN),
        OtherN > NColors
    ).
    