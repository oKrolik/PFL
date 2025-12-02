list_nth(N, List, E) :-
    append(L1, [E|_], List),
    length(L1, N).
    