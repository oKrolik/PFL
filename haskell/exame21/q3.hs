transforma :: String -> String
transforma [] = []
transforma (x:xs)
    | x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' = x : 'p' : x : transforma xs
    | otherwise = x : transforma xs