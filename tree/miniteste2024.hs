-- (("Porto", "Benfica"), (7,0))

type Match = ((String, String), (Int, Int))
type MatchDay = [Match]
type League = [MatchDay]

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
-- bigWins :: League -> [(Int, [String])]