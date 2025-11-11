max3_a, min3_a :: Ord a => a -> a -> a -> a
max3_a x y z = if x>=y && x>=z then x 
               else if y>=z then y 
               else z
min3_a x y z = if x<=y && x<=z then x 
               else if y<=z then y
               else z


max3_b, min3_b :: Ord a => a -> a -> a -> a
max3_b x y z = max x (max y z)
min3_b x y z = min x (min y z)