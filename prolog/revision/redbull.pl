% Pilots
pilot(lamb).
pilot(besenyei).
pilot(chambliss).
pilot(maclean).
pilot(mangold).
pilot(jones).
pilot(bonhomme).

% Teams
team(lamb, breitling).
team(besenyei, red_bull).
team(chambliss, red_bull).
team(maclean, mediterranean_racing_team).
team(mangold, cobra).
team(jones, matador).
team(bonhomme, matador).

% Flies
fly(lamb, mx2).
fly(besenyei, edge540).
fly(chambliss, edge540).
fly(maclean, edge540).
fly(mangold, edge540).
fly(jones, edge540).
fly(bonhomme, edge540).
% circuits
circuit(istanbul).
circuit(budapest).
circuit(porto).
% Winners
winner(jones, porto).
winner(mangold, budapest).
winner(mangold, istanbul).
% Gates
gates(istanbul, 9).
gates(budapest, 6).
gates(porto, 5).

% Winner teams
winner_team(Team, Race) :-
    winner(Pilot, Race),
    team(Pilot, Team).

does_not_fly_edge540(Pilot) :-
    pilot(Pilot),
    \+ fly(Pilot, edge540).

won_more_than_one_circuit(Pilot) :-
    winner(Pilot, Circuit1),
    winner(Pilot, Circuit2),
    Circuit1 \= Circuit2.

plane_that_won_porto(Plane) :-
    fly(Pilot, Plane),
    winner(Pilot, porto).