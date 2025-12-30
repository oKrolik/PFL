% ===================================================================
%% Uluru Puzzle Solver
%% ===================================================================

%% ------------------------------------------------------------------
%% Define as cores disponiveis no puzzle
colors([green, yellow, blue, orange, white, black]).
%% ------------------------------------------------------------------



% ===================================================================
%% Predicados de restrição
% ===================================================================

%% ------------------------------------------------------------------
%% anywhere(+Color, +Board):
%% Color pode estar em qualquer posicao (sempre satisfeita)
%% ------------------------------------------------------------------
anywhere(_, _).



%% ------------------------------------------------------------------
%% next_to(+X, +Y, +Board):
%% X deve estar ao lado de Y
%% ------------------------------------------------------------------
next_to(X, X, _).
next_to(X, Y, Board) :-
    consecutive(X, Y, Board).
next_to(X, Y, Board) :-
    consecutive(Y, X, Board).



%% ------------------------------------------------------------------
%% consecutive(+X, +Y, +Board):
%% X e Y ocorrem consecutivamente em Board
%% ------------------------------------------------------------------
consecutive(X, Y, Board) :-
    append(_, [X, Y | _], Board).



%% ------------------------------------------------------------------
%% one_space(X, Y, Board):
%% X deve estar com um espaco entre ele e Y
%% ------------------------------------------------------------------
one_space(X, X, _).
one_space(X, Y, Board) :-
    skip_one(X, Y, Board).
one_space(X, Y, Board) :-
    skip_one(Y, X, Board).



%% ------------------------------------------------------------------
%% skip_one(+X, +Y, +Board):
%% X e Y estao separados por um espaco em Board
%% ------------------------------------------------------------------
skip_one(X, Y, Board) :-
    append(_, [X, _, Y | _], Board).



%% ------------------------------------------------------------------
%% across(X, Y, Board):
%% X deve estar do outro lado de Y
%% ------------------------------------------------------------------
%% across_indices(+IX, +IY):
%% Auxiliar para across/3 – verifica se IX e IY estao em posicoes opostas
%% ------------------------------------------------------------------
across(X, X, _).
across(X, Y, Board) :-
    position_index(X, Board, IX),
    position_index(Y, Board, IY),
    across_indices(IX, IY).

across_indices(I, J) :-
    member(I, [1,2]),
    member(J, [4,5,6]).
across_indices(I, J) :-
    member(I, [4,5,6]),
    member(J, [1,2]).



%% ------------------------------------------------------------------
%% same_edge(X, Y, Board):
%% X deve estar na mesma borda que Y
%% ------------------------------------------------------------------
%% same_edge_indices(+IX, +IY):
%% Auxiliar para same_edge/3 – verifica se IX e IY estao na mesma borda
%% ------------------------------------------------------------------
same_edge(X, X, _).
same_edge(X, Y, Board) :-
    position_index(X, Board, IX),
    position_index(Y, Board, IY),
    same_edge_indices(IX, IY).

same_edge_indices(I, J) :-
    member(I, [1,2]),
    member(J, [1,2]).
same_edge_indices(I, J) :-
    member(I, [4,5,6]),
    member(J, [4,5,6]).



%% ------------------------------------------------------------------
%% position(X, L, Board):
%% X deve estar em uma das posicoes dadas na lista L
%% ------------------------------------------------------------------
%% position_index(+Colour, +Board, -Index):
%% True quando Colour aparece na posicao Index (1-based) em Board.
%% ------------------------------------------------------------------
position(X, L, Board) :-
    position_index(X, Board, I),
    member(I, L).

%% position_index(+Colour, +Board, -Index)
%% True when Colour appears at 1‑based position Index in Board.
position_index(Color, Board, Index) :-
    nth1_custom(Index, Board, Color).



% ===================================================================
%% Predicados principais
% ===================================================================

%% ------------------------------------------------------------------
%% perm(+List, -Perm):
%% Gera uma permutacao Perm da lista List.
%% ------------------------------------------------------------------
perm([], []).
perm(List, [X | Perm]) :-
    select_custom(X, List, Rest),
    perm(Rest, Perm).



%% ------------------------------------------------------------------
%% select_custom(?Elem, +List, -Rest):
%% Remove uma ocorrencia de Elem de List, resultando em Rest.
%% ------------------------------------------------------------------
select_custom(X, [X|Xs], Xs).
select_custom(X, [Y|Ys], [Y|Rest]) :-
    select_custom(X, Ys, Rest).



%% ------------------------------------------------------------------
%% nth1_custom(?Index, +List, ?Elem):
%% Elemento na posicao Index (1-based) de List.
%% ------------------------------------------------------------------
nth1_custom(Index, List, Elem) :-
    nth1_custom_(List, 1, Index, Elem).

nth1_custom_([H|_], I, Index, H) :-
    Index = I.
nth1_custom_([_|T], I, Index, Elem) :-
    I1 is I + 1,
    nth1_custom_(T, I1, Index, Elem).



%% ------------------------------------------------------------------
%% max_list_custom(+List, -Max):
%% Retorna o maior elemento de List.
%% ------------------------------------------------------------------
max_list_custom([H|T], Max) :-
    max_list_custom(T, H, Max).

max_list_custom([], M, M).
max_list_custom([H|T], M0, M) :-
    ( H >= M0 -> M1 = H ; M1 = M0 ),
    max_list_custom(T, M1, M).



%% ------------------------------------------------------------------
%% solve(+Constraints, -Board):
%% Encontra uma solucao Board que satisfaz todas as Constraints.
%% ------------------------------------------------------------------
%% satisfies_all(+Constraints, +Board):
%% Verifica se todas as Constraints sao satisfeitas em Board.
%% ------------------------------------------------------------------
solve(Constraints, Board) :-
    colors(Colors),
    perm(Colors, Board),
    satisfies_all(Constraints, Board).

satisfies_all([], _).
satisfies_all([Constraint | Rest], Board) :-
    call(Constraint, Board),
    satisfies_all(Rest, Board).



%% ------------------------------------------------------------------
%% best_score(+Constraints, -Score):
%% Computa a melhor pontuacao para uma lista de constraints.
%% ------------------------------------------------------------------
%% count_satisfied(+Constraints, +Board, -Count):
%% Conta quantas Constraints sao satisfeitas em Board.
%% ------------------------------------------------------------------
best_score(Constraints, Score) :-
    length(Constraints, K),
    colors(Colors),
    findall(Count,
            (   perm(Colors, Board),
                count_satisfied(Constraints, Board, Count)
            ),
            Counts),
    max_list_custom(Counts, MaxSatisfied),
    Score is MaxSatisfied - K.

count_satisfied([], _, 0).
count_satisfied([Constraint | Rest], Board, Count) :-
    call(Constraint, Board),
    !,
    count_satisfied(Rest, Board, CountRest),
    Count is CountRest + 1.
count_satisfied([_ | Rest], Board, Count) :-
    count_satisfied(Rest, Board, Count).



% ===================================================================
%% Example puzzles
% ===================================================================

%% ------------------------------------------------------------------
%% example(1, Constraints):
%% Esse puzzle tem 12 soluções.
%% ------------------------------------------------------------------
example(1, [ next_to(white, orange),
             next_to(black, black),
             across(yellow, orange),
             next_to(green, yellow),
             position(blue, [1,2,6]),
             across(yellow, blue) ]).



%% ------------------------------------------------------------------
%% example(2, Constraints):
%% exatamente uma solução.
%% ------------------------------------------------------------------
example(2, [ across(white, yellow),
             position(black, [1,4]),
             position(yellow, [1,5]),
             next_to(green, blue),
             same_edge(blue, yellow),
             one_space(orange, black) ]).



%% ------------------------------------------------------------------
%% example(3, Constraints):
%% não há solução completa; 5 das 6 constraints podem ser satisfeitas simultaneamente.
%% ------------------------------------------------------------------
example(3, [ across(white, yellow),
             position(black, [1,4]),
             position(yellow, [1,5]),
             same_edge(green, black),
             same_edge(blue, yellow),
             one_space(orange, black) ]).



%% ------------------------------------------------------------------
%% example(4, Constraints):
%% Igual ao exemplo(3) mas com as
%% constraints listadas em uma ordem diferente. O resultado para
%% best_score/2 deve ser o mesmo.
%% ------------------------------------------------------------------
example(4, [ position(yellow, [1,5]),
             one_space(orange, black),
             same_edge(green, black),
             same_edge(blue, yellow),
             position(black, [1,4]),
             across(white, yellow) ]).




% ===================================================================
%% Predicados de teste
% ===================================================================

%% ------------------------------------------------------------------
%% run_example(+N):
%% Executa o exemplo N, imprimindo todas as soluções encontradas.
%% ------------------------------------------------------------------
run_example(N) :-
    example(N, Constraints),
    solve(Constraints, Board),
    write(Board), nl,
    fail.
run_example(_).

% ===================================================================
%% Extra
% ===================================================================
