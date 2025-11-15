-- (("Porto", "Benfica"), (7,0))

type Match = ((String, String), (Int, Int))
type MatchDay = [Match]
type League = [MatchDay]

myLeague :: League
myLeague = [[(("Porto","Sporting"),(2,2)),(("Benfica","Vitoria SC"),(4,0))],[(("Porto","Benfica"),(5,0)),(("Vitoria SC","Sporting"),(3,2))],[(("Vitoria SC","Porto"),(1,2)),(("Sporting","Benfica"),(2,1))]]


-- "Porto" ou "Draw"
winner :: Match -> String
winner ((t1, t2), (s1, s2))
    | s1 > s2 = t1
    | s2 > s1 = t2
    | otherwise = "Draw"

-- 3 se ganhou, 1 se empatou, 0 se perdeu
matchDayScore :: String -> MatchDay -> Int
matchDayScore team xs = sum [score team matchs | matchs <- xs]
    where
        score t ((t1, t2), (s1, s2))
            | winner ((t1, t2), (s1, s2)) == t && t == t1 = 3
            | winner ((t1, t2), (s1, s2)) == t && t == t2 = 3
            | winner ((t1, t2), (s1, s2)) == "Draw" && (t == t1 || t == t2) = 1
            | otherwise = 0

numMatchDaysWithDraws :: League -> Int
numMatchDaysWithDraws league = length [() | matchDay <- league, match <- matchDay, winner match == "Draw"]


-- Lista das equipas que ganharam por muito (+3 gols de vantagens)
-- [(1, ["Porto"]), (2, ["Porto", "Benfica"]), ...]
bigWins :: League -> [(Int, [String])]
bigWins league = [(i, [winner m | m@((t1, t2), (s1, s2)) <- matchDay, abs (s1 - s2) >= 3]) | (i, matchDay) <- zip [1..] league]
