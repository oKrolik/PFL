% Note: parent(a, b) is interpreted as 'a is a parent of b'.

female(grace).
male(frank).
female(dede).
male(jay).
female(gloria).
male(javier).
female(barb).
male(merle).
male(phill).
female(claire).
male(mitchell).
male(joe).
female(manny).
female(cameron).
male(pameron).
female(bo).
male(dylan).
female(haley).
male(alex).
male(luke).
female(lily).
male(rexford).
male(calhoun).
male(george).
female(poppy).
parent(grace, phil).
parent(frank, phil).
parent(dede, claire).
parent(jay, claire).
parent(dede, mitchell).
parent(jay, mitchell).
parent(jay, joe).
parent(gloria, joe).
parent(gloria, manny).
parent(javier, manny).
parent(barb, cameron).
parent(merle, cameron).
parent(barb, pameron).
parent(merle, pameron).
parent(phill, haley).
parent(claire, haley).
parent(phill, alex).
parent(claire, alex).
parent(phill, luke).
parent(claire, luke).
parent(mitchell, lily).
parent(cameron, lily).
parent(mitchell, rexford).
parent(cameron, rexford).
parent(pameron, calhoun).
parent(bo, calhoun).
parent(dylan, george).
parent(haley, george).
parent(dylan, poppy).
parent(haley, poppy).

father(X, Y) :-
    parent(X, Y),
    male(X).

mother(X, Y) :-
    parent(X, Y),
    female(X).

% siblings/2 checks if X and Y share both parents Z and K, 
% and ensures that X and Y are not the same individual.
siblings(X, Y) :- 
    parent(Z, X), 
    parent(Z, Y), 
    parent(K, X), 
    parent(K, Y), 
    Z \= K, 
    X \= Y.

% halfSiblings/2 checks if X and Y share a parent Z, 
% and that Z is not a sibling of another parent K. 
halfsiblings(X, Y) :- 
    parent(Z, X), 
    parent(Z, Y), 
    Z \= K, 
    \+ siblings(Z, K).

% grandparent/2 checks if X is a parent of Z, 
% and Z is a parent of Y, thus making X a grandparent of Y.
grandparent(X, Y) :- 
    parent(X, Z), 
    parent(Z, Y).

% to get jay's grandchildren -> parent(jay, _x),parent(_x, y).
% to get jdoe's grandparents -> parent(y, _x),parent(_x, jdoe).
