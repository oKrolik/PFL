{-

1.7 Mark all the following expressions that have a type error.
(a) 1 + 1.5
(b) 1 + False               -- Cannot add Int and Bool
(c) ’a’ + ’b’               -- Cannot add Chars
(d) ’a’ ++ ’b’              -- ++ expects [Char], not Char
(e) "a" ++ "b"
(f) "1+2" == "3"
(g) 1+2 == "3"              -- Cannot compare Int and String
(h) show (1+2) == "3"
(i) ’a’ < ’b’
(j) ’a’ < "ab"              -- Cannot compare Char and String
(k) (1 <= 2) <= 3           -- Cannot compare Bool and Int
(l) (1 <= 2) < (3 <= 4)
(m) head [1,2]
(n) head (1,2)              -- head expects a list, not a tuple
(o) tail "abc"

-}