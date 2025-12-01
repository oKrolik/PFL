pow(X, 0, 1).
pow(X, 1, X).
pow(X, Y, P) :-
    Y1 is Y - 1,
    pow(X, Y1, P1),
    P is X * P1.