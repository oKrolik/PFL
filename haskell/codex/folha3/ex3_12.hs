intersperse :: a -> [a] -> [a]
intersperse _ [] = []
intersperse _ [x] = [x]
intersperse s (x:xs) = x : s : intersperse s xs


-- intersperse '-' "banana" == "b-a-n-a-n-a"
-- intersperse 0 [1,2,3] == [1,0,2,0,3]
-- intersperse 0 [1] == [1]
-- intersperse 0 []  == []