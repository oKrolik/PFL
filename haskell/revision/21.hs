maxpos :: [Int] -> Int
maxpos [] = 0
maxpos [x] = x
maxpos (x:xs)
    | x > next = x
    | otherwise = next
    where
        next = maxpos xs

-- maxpos [1,2,3,4,5] == 5
-- maxpos [-1,-2,-3,4,-5] == 4
-- maxpos [2] == 2
-- maxpos [] == 0

dups :: [a] -> [a]
dups [] = []
dups [x] = [x,x]
dups [x,y] = [x,x,y]
dups (x:y:xs) = (x:x:y:dups xs)

-- dups "abcdef" == "aabccdeef"
-- dups [0,1,2,3,4] == [0,0,1,2,2,3,4,4]
-- dups [] == []

transforma :: String -> String
transforma [] = []
transforma (x:xs)
    | x `elem` "aeiou" = x : 'p' : x : transforma xs
    | otherwise       = x : transforma xs


-- transforma "ola, mundo!" == "opolapa, mupundopo!"
-- transforma "4 gatos e 3 ratos" == "4 gapatopos epe 3 rapatopos"

type Vector = [Int]
type Matriz = [[Int]]
-- Nota: as matrizes são retangulares, ou seja, o comprimento de todas as sublistas é idêntico.

prodInterno :: Vector -> Vector -> Int
prodInterno [] [] = 0
prodInterno (x:xs) (y:ys) = x * y + prodInterno xs ys

-- prodInterno [1,2,3] [4,3,2] = 16

type Species = (String, Int)
type Zoo = [Species]

isEndangered :: Species -> Bool
isEndangered (name, count)
    | count <= 100 = True
    | otherwise = False

updateSpecies :: Species -> Int -> Species
updateSpecies (name, oldCount) newCount = (name, newCount)

type Node = String   -- some city
type Dist = Int      -- some distance
type Edges = [(Node,Node,Dist)] -- directed connections

portugal :: Edges
portugal = [ ("Porto", "Aveiro", 76)
           , ("Aveiro", "Coimbra", 63)
           , ("Aveiro", "Leiria", 117)
           , ("Coimbra", "Leiria", 76)
           , ("Leiria", "Santarem", 83)
           , ("Santarem", "Lisboa", 82)
           ]

inverse :: Edges -> Edges
inverse edges = [(c1, c2, d) | (c2, c1, d) <- edges]

-- ghci> inverse [("A","B",10), ("C","D",15)]
-- [("B","A",10), ("D","C",15)]

newEdges :: Edges -> Edges
newEdges edges = [(ci1, cj2, d1 + d2) | (ci1, cj1, d1) <- edges, (ci2, cj2, d2) <- edges, ci2==cj1 && ci1/=cj2]

-- ghci> newEdges [("A","B",10), ("B","C",15), ("C","D",20)]
-- [("A","C",25), ("B","D",35)]
-- ghci> newEdges [("A","B",10), ("B","A",15)]
-- []

--pathDistance :: Edges -> [Node] -> Maybe Dist