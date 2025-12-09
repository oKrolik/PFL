count(E, [], 0).

count(H, [H|T], N):-
    count(H, T, N1),
    N is N1 + 1.

count(E, [H|T], N):-
    E \= H,
    count(E, T, N).