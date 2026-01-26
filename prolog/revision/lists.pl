% list_size(+List, ?Size)
list_size([], 0).
list_size([_H | T], Size) :-
    list_size(T, SizeT),
    Size is SizeT + 1.

% list_sum(+List, ?Sum)
list_sum([], 0).
list_sum([H], H).
list_sum([H | T], Sum) :-
    list_sum(T, SumT),
    Sum is SumT + H.

% list_prod(+List, ?Prod)
list_prod([], 0).
list_prod([H], H).
list_prod([H | T], Prod) :-
    list_prod(T, ProdT),
    Prod is ProdT * H.

invert([], []).
invert([H | T], Inverted) :-
    invert(T, InvertedT),
    append(InvertedT, [H], Inverted).