before(F, S, L) :-
    append(_, [F|L2], L),
    append(_, [S|_], L2).