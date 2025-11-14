binom :: Integer -> Integer -> Integer
binom n k = fact n `div` (fact k * (fact (n-k)))

fact :: Integer -> Integer
fact n = product [1..n]

pascal :: Integer -> [[Integer]]
pascal x = [ [ binom n k | k <- [0..n] ] | n <- [0..x] ]