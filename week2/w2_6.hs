short :: [a] -> Bool
short xs = length xs <= 3

shortPatterns :: [a] -> Bool
shortPatterns [] = True
shortPatterns [a] = True
shortPatterns [a,b] = True
shortPatterns [a,b,c] = True
shortPatterns (x:xs) = False