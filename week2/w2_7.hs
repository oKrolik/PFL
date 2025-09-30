max3, min3 :: (Ord a, Num a) => a -> a -> a -> a
max3 x y z
    | x>=y && x>=z = x
    | y>=z = y 
    | otherwise = z
min3 x y z
    | x<=y && x<=z = x
    | y<=z = y
    | otherwise = z

median :: (Ord a, Num a) => a -> a -> a -> a
median x y z = x+y+z - max3 x y z - min3 x y z