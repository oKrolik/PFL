mediana, max3, min3 :: Int -> Int -> Int -> Int
mediana a b c = a + b + c - (max3 a b c) - (min3 a b c)

max3 a b c = max a (max b c)
min3 a b c = min a (min b c)