myAnd :: [Bool] -> Bool
myAnd [] = True
myAnd (x:xs) = x && myAnd xs

myOr :: [Bool] -> Bool 
myOr [] = False
myOr (x:xs) = x || myOr xs

myConcat :: [[a]] -> [a]
myConcat [] = []
myConcat (x:xs) = x ++ myConcat xs

myReplicate :: Int -> a -> [a]
myReplicate 0 x = []
myReplicate n x | n > 0 = x:myReplicate (n-1) x

(!!!) :: [a] -> Int -> a
(!!!) (x:xs) 0 = x
(!!!) (x:xs) n | n > 0 = (!!!) xs (n-1)

myElem :: Eq a => a -> [a] -> Bool
myElem n [] = False
myElem y (x:xs) = (y == x) || myElem y xs