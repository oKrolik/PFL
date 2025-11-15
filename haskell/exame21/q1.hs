maxpos :: [Int] -> Int
maxpos [] = 0
maxpos (x:xs)
    | x > next = x
    | otherwise = next
    where
        next = maxpos xs