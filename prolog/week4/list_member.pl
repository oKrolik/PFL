% Verifies if E is on List, like member(E ,List)

list_member(E, List) :-
    append(_, [E|_], List).