propDivs :: Integer -> [Integer]
propDivs n = [x | x <- [1..n-1], n `mod` x == 0]

perfects :: Integer -> [Integer]
perfects n = [x | x <- [1..n], sum (propDivs x) == x]