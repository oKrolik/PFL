% factorial(+N, -F)
factorial(0, 1).
factorial(1, 1).
factorial(N, F) :-
    N > 1,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.

% sum_rec(+N, -Sum)
sum_rec(0, 0).
sum_rec(1, 1).
sum_rec(N, S) :-
    N > 1,
    N1 is N - 1,
    sum_rec(N1, S1),
    S is N + S1.
    
% has_factor(+N, +F)
has_factor(N, F) :-
    N mod F =:= 0.

% is_prime(+N)
is_prime(2).
is_prime(N) :-
    N > 2,
    \+ has_factor(N, 2).