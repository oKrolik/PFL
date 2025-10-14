insert :: Ord a => a -> [a] -> [a]
insert x [] = [x]
insert x (y:ys) | y>=x = x:y:ys
                | otherwise = y:insert x ys

isort :: Ord a => [a] -> [a]
isort (x:xs) = insert x (isort xs)
isort [] = []