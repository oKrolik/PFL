calcPi1,calcPi2 :: Int -> Double
calcPi1 n = sum (take n (zipWith (/) (cycle [4,-4]) [1,3..]))
calcPi2 n = 3.0 + sum (take n (zipWith (/) (cycle [4,-4]) [k * (k+1) * (k+2) | k <- [2,4..]]))