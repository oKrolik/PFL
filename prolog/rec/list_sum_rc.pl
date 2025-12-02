% list_sum_rc(List, Sum)

list_sum_rc(List, Sum):-
    l_s(List, 0, Sum).

l_s([], Acc, Acc).
l_s([H|T], Acc, Sum):-
    Acc1 is H + Acc,
    l_s(T, Acc1, Sum).