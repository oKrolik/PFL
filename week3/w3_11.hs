(+++) :: [a] -> [a] -> [a]
(+++) xs ys = foldr (:) ys xs

myconcat :: [[a]] -> [a]
myconcat xs = foldr (++) [] xs
