% list_sum(List, Sum)

list_sum([], 0).
list_sum([H|T], Sum) :-
    list_sum(T, Sum1),
    Sum is H + Sum1.