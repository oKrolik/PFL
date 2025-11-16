type Vector = [Int]
type Matriz = [[Int]]

prodInterno :: Vector -> Vector -> Int
prodInterno [] [] = 0
prodInterno (x:xs) (y:ys) = x * y + prodInterno xs ys