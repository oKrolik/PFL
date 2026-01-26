%author(AuthorID, Name, YearOfBirth, CountryOfBirth).
author(1, 'John Grisham', 1955, 'USA').
author(2, 'Wilbur Smith', 1933, 'Zambia').
author(3, 'Stephen King', 1947, 'USA').
author(4, 'Michael Crichton', 1942, 'USA').
%book(Title, AuthorID, YearOfRelease, Pages, Genres).
book('The Firm', 1, 1991, 432, ['Legal thriller']).
book('The Client', 1, 1993, 422, ['Legal thriller']).
book('The Runaway Jury', 1, 1996, 414, ['Legal thriller']).
book('The Exchange', 1, 2023, 338, ['Legal thriller']).
book('Carrie', 3, 1974, 199, ['Horror']).
book('The Shining', 3, 1977, 447, ['Gothic novel', 'Horror', 'Psychological horror']).
book('Under the Dome', 3, 2009, 1074, ['Science fiction', 'Political']).
book('Doctor Sleep', 3, 2013, 531, ['Horror', 'Gothic', 'Dark fantasy']).
book('Jurassic Park', 4, 1990, 399, ['Science fiction']).
book('Prey', 4, 2002, 502, ['Science fiction', 'Techno-thriller', 'Horror', 'Nanopunk']).
book('Next', 4, 2006, 528, ['Science fiction', 'Techno-thriller', 'Satire']).

% book_author(?Title, ?Author)
book_author(Title, Author) :-
    book(Title, AuthorID, _, _, _),
    author(AuthorID, Author, _, _).

% multi_genre_book(?Title)
multi_genre_book(Title) :-
    book(Title, _, _, _, [_H, _M | T]).

% shared_genres(?Title1, ?Title2, -Shared)
comp([], _Genre2, []).
comp([H | T], Genre2, [H | Common]) :-
    member(H, Genre2), !,
    comp(T, Genre2, Common).

comp([_ | T], Genre2, Common) :-
    comp(T, Genre2, Common).

shared_genres(Title1, Title2, Shared) :-
    book(Title1, _, _, _, G1),
    book(Title2, _, _, _, G2),
    comp(G1, G2, Shared).


% similarity(?Title1, ?Title2, ?Similarity)
similarity(Title1, Title2, Sim) :-
    book(Title1, _, _, _, G1),
    book(Title2, _, _, _, G2),
    shared_genres(Title1, Title2, Intersection),
    length(Intersection, IntLen),
    append(G1, G2, List),
    length(List, ListLen),
    UniLen is ListLen - IntLen,
    Sim is IntLen / UniLen.
    


% gives_gift_to(Giver, Gift, Receiver)
gives_gift_to(bernardete, 'The Exchange', celestina).
gives_gift_to(celestina, 'The Brethren', eleuterio).
gives_gift_to(eleuterio, 'The Summons', felismina).
gives_gift_to(felismina, 'River God', juvenaldo).
gives_gift_to(juvenaldo, 'Seventh Scroll', leonilde).
gives_gift_to(leonilde, 'Sunbird', bernardete).
gives_gift_to(marciliano, 'Those in Peril', nivaldo).
gives_gift_to(nivaldo, 'Vicious Circle', sandrino).
gives_gift_to(sandrino, 'Predator', marciliano).

interact(Person, Person).
interact([H | T], People):-
    gives_gift_to(H, _, NewPerson),
    \+ member(NewPerson, [H | T]), !,
    interact([NewPerson, H | T], People).

circle_size(Person, Size):-
    interact([Person], People),
    length(People, Size).
