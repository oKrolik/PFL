reversefl :: [a] -> [a]
reversefl xs = foldl (\acc x -> [x] ++ acc) [] xs

reversefr :: [a] -> [a]
reversefr xs = foldr (\x acc -> acc ++ [x]) [] xs