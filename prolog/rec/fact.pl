fact(0,1).
fact(N,F) :- N > 0, N1 is N - 1, fact(N1,F1), F is N * F1.


fact2(N,F) :-
    factAux(N, 1, F).

factAux(0, Acc, Acc).
factAux(N, Acc, F) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factAux(N1, Acc1, F).