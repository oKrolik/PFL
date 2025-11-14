myand, myor :: [Bool] -> Bool
myand [] = True
myand (x:xs)
    | x == False = False
    | otherwise = myand xs

myor [] = False
myor (x:xs)
    | x == True = True
    | otherwise = myor xs

