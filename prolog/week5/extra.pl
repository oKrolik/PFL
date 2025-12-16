teachers(-L):-
    findall(T, teachers(T, C), L),
    sort(UL, L).

stud_of(+T, -S)