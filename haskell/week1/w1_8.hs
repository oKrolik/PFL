{-

1.8 Determine the most general type for each of the following definitions. You
should include type class restrictions for any overloaded operations.

(a) second xs = head (tail xs)
    second :: [a] -> a

(b) swap (x,y) = (y,x)
    swap :: (a, b) -> (b, a)

(c) pair x = (x,x)
    pair :: a -> (a, a)

(d) double x = 2*x
    double :: Num a => a -> a

(e) half x = x/2
    half :: Fractional a => a -> a

(f) average x y = (x+y)/2
    average :: Fractional a => a -> a -> a

(g) isLower x = x >= 'a' && x <= 'z'
    isLower :: Char -> Bool

(h) inRange x lo hi = x >= lo && x <= hi
    inRange :: Ord a => a -> a -> a -> Bool

(i) isPalindrome xs = xs == reverse xs
    isPalindrome :: Eq a => [a] -> Bool

(j) twice f x = f (f x)
    twice :: (a -> a) -> a -> a

-}