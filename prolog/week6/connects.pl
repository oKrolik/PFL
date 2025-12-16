c_t(a,b).
c_t(a,c).
c_t(b,c).
c_t(b,d).
c_t(c,e).
c_t(c,f).
c_t(d,c).
c_t(d,f).
c_t(e,a).
c_t(e,f).
c_t(f,d).
c_t(f,e).

connects(X, Y, F) :-
    connects(X, [X], Y, F).

connects(X, _P, X, F) :-
    reverse(_P, F).

connects(X, P, Y, F) :-
    c_t(X, Z),
    \+ member(Z, P),
    connects(Z, [Z|P], Y, F).

reverse(L, R) :- reverse_acc(L, [], R).

reverse_acc([], Acc, Acc).
reverse_acc([H|T], Acc, R) :-
    reverse_acc(T, [H|Acc], R).
