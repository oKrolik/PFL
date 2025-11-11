binom :: Integer -> Integer -> Integer
binom n k = fact n `div` (fact k * (fact (n-k)))

fact :: Integer -> Integer
fact n = product [1..n]