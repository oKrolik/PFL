# Haskell
## 1. CORE LIST OPERATIONS
```hs
myConcat :: [[a]] -> [a]
myConcat = foldr (++) []

myReplicate :: Int -> a -> [a]
myReplicate n x
    | n > 0 = x : myReplicate (n-1) x
    | otherwise = []

(!!!) :: [a] -> Int -> a
(x:xs) !!! 0 = x
(x:xs) !!! n = xs !!! (n-1)

myElem :: Eq a => a -> [a] -> Bool
myElem _ [] = False
myElem y (x:xs) = (y == x) || myElem y xs

(+++) :: [a] -> [a] -> [a]
(+++) xs ys = foldr (:) ys xs

-- Remove duplicates
nub :: Eq a => [a] -> [a]
nub [] = []
nub (x:xs) = x : nub (filter (/= x) xs)

-- Same but using the supplied argument function to check for equality
nubBy :: (a -> a -> Bool) -> [a] -> [a]
nubBy eq [] = []
nubBy eq (x:xs) = x : nubBy eq [x' | x'<-xs, not (eq x x')]

-- Intersperse element between list elements
intersperse :: a -> [a] -> [a]
intersperse _ [] = []
intersperse _ [x] = [x]
intersperse s (x:xs) = x : s : intersperse s xs

-- Reverse implementations
reversefl, reversefr :: [a] -> [a]
reversefl = foldl (flip (:)) []
reversefr = foldr (\x acc -> acc ++ [x]) []
```
## 2. FOLD PATTERNS
```hs
sumList = foldr (+) 0
productList = foldr (*) 1
andList = foldr (&&) True
orList = foldr (||) False
concatList = foldr (++) []
lengthList = foldr (\_ acc -> acc + 1) 0
maximumList = foldr1 max
minimumList = foldr1 min
```
## 3. SORTING & SEARCHING
```hs
-- Insertion sort
isort :: Ord a => [a] -> [a]
isort [] = []
isort (x:xs) = insert x (isort xs)

insert :: Ord a => a -> [a] -> [a]
insert x [] = [x]
insert x (y:ys) | x <= y = x : y : ys
                | otherwise = y : insert x ys

-- Merge two sorted lists
merge :: Ord a => [a] -> [a] -> (a -> a -> Bool) -> [a]
merge [] l _ = l
merge l [] _ = l
merge (x:xs) (y:ys) cmp
    | cmp x y = x : merge xs (y:ys) cmp
    | otherwise = y : merge (x:xs) ys cmp

-- Merge sort
sortByCond :: Ord a => [a] -> (a -> a -> Bool) -> [a]
sortByCond [] _ = []
sortByCond [x] _ = [x]
sortByCond l cmp = merge (sortByCond l1 cmp) (sortByCond l2 cmp) cmp
where (l1, l2) = splitAt (length l `div` 2) l
```
## 4. HIGHER-ORDER FUNCTIONS
```hs
-- Composition & application
(.) :: (b -> c) -> (a -> b) -> a -> c
(f . g) x = f (g x)

($) :: (a -> b) -> a -> b
f $ x = f x

-- Function application patterns
twice :: (a -> a) -> a -> a
twice f x = f (f x)

-- Map, filter, zipWith patterns
map f [] = []
map f (x:xs) = f x : map f xs

filter p [] = []
filter p (x:xs) | p x = x : filter p xs
                | otherwise = filter p xs

zipWith f [] _ = []
zipWith f _ [] = []
zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys
```
## 5. TREES & RECURSIVE DATA STRUCTURES
```hs
data Arv a = F | N a (Arv a) (Arv a) deriving Show
data Set a = Empty | Node a (Set a) (Set a)

-- BST Insert
insert :: Ord a => a -> Set a -> Set a
insert x Empty = Node x Empty Empty
insert x (Node y left right)
    | x < y = Node y (insert x left) right
    | x > y = Node y left (insert x right)
    | otherwise = Node y left right

-- BST Member
member :: Ord a => a -> Set a -> Bool
member _ Empty = False
member x (Node y left right)
    | x == y = True
    | x < y = member x left
    | otherwise = member x right

-- Tree height
height :: Arv a -> Int
height F = 0
height (N _ l r) = 1 + max (height l) (height r)

-- Tree map
treeMap :: (a -> b) -> Set a -> Set b
treeMap _ Empty = Empty
treeMap f (Node v l r) = Node (f v) (treeMap f l) (treeMap f r)

-- Build balanced BST from sorted list
fromList :: Ord a => [a] -> Set a
fromList xs = build (sort xs)
where build [] = Empty
    build xs = Node x (build xs') (build xs'')
    where k = length xs `div` 2
        xs' = take k xs
        (x:xs'') = drop k xs
```
## 6. MATRICES & LINEAR ALGEBRA
```hs
type Vector = [Int]
type Matriz = [[Int]]

transposta :: Matriz -> Matriz
transposta [] = []
transposta m = [head x | x <- m] : transposta [tail x | x <- m, tail x /= []]

prodInterno :: Vector -> Vector -> Int
prodInterno [] [] = 0
prodInterno (x:xs) (y:ys) = x * y + prodInterno xs ys

prodMat :: Matriz -> Matriz -> Matriz
prodMat m1 m2 = [[prodInterno v1 v2 | v2 <- transposta m2] | v1 <- m1]
```
## 7. COMBINATORICS & MATH
```hs
fact :: Integer -> Integer
fact n = product [1..n]

binom :: Integer -> Integer -> Integer
binom n k = fact n `div` (fact k * fact (n-k))

pascal :: Integer -> [[Integer]]
pascal x = [[binom n k | k <- [0..n]] | n <- [0..x]]

propDivs :: Integer -> [Integer]
propDivs n = [x | x <- [1..n-1], n `mod` x == 0]

perfects :: Integer -> [Integer]
perfects n = [x | x <- [1..n], sum (propDivs x) == x]

pyths :: Integer -> [(Integer,Integer,Integer)]
pyths n = [(x,y,z) | x <- [1..n], y <- [1..n], z <- [1..n], x^2 + y^2 == z^2]
```
## 8. INFINITE LISTS & LAZY EVALUATION
```hs
-- Fibonacci
fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

-- Primes (basic sieve)
primes :: [Integer]
primes = sieve [2..]
where sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]

-- Twin primes
twinPrimes :: [(Integer,Integer)]
twinPrimes = filter (\(x,y) -> y == x+2) (zip primes (tail primes))

-- Recursive series: a_n = 3*a_(n-1) - 2*a_(n-2)
series :: [Integer]
series = 0 : 1 : zipWith (\a1 a2 -> 3*a1 - 2*a2) (tail series) series

-- Pi approximations
calcPi1, calcPi2 :: Int -> Double
calcPi1 n = sum (take n (zipWith (/) (cycle [4,-4]) [1,3..]))
calcPi2 n = 3.0 + sum (take n (zipWith (/) (cycle [4,-4]) [k*(k+1)*(k+2) | k <- [2,4..]]))
```
## 9. STRING & CHAR OPERATIONS
```hs
-- Password strength
forte :: String -> Bool
forte str = length str >= 8 && any (`elem` ['A'..'Z']) str && any (`elem` ['a'..'z']) str && any (`elem` ['0'..'9']) str
```
## 10. TYPE CLASSES & TYPE INFERENCE
```hs
-- Common type signatures
second :: [a] -> a
swap :: (a, b) -> (b, a)
pair :: a -> (a, a)
double :: Num a => a -> a
half :: Fractional a => a -> a
average :: Fractional a => a -> a -> a
isLower :: Char -> Bool
inRange :: Ord a => a -> a -> a -> Bool
isPalindrome :: Eq a => [a] -> Bool
twice :: (a -> a) -> a -> a

-- Type class hierarchy
-- Eq -> Ord -> Num -> Integral
--                 \-> Fractional -> Floating

-- "type" vs "data":
-- - Only "data" allows recursive definitions
-- - Only "data" defines new patterns for pattern matching
-- - "data" can derive typeclasses like Eq, Show, etc.
```
## 11. MONADS & IO
```hs
-- Maybe, State, IO are all monads
-- IO type signature example:
-- putStrLn :: String -> IO ()
-- getLine :: IO String

-- Do notation pattern:
readAndProcess :: IO ()
readAndProcess = do
    x <- getLine
    if x == "quit"
        then return ()
        else do
            putStrLn ("You said: " ++ x)
            readAndProcess
```
## 12. GRAPH ALGORITHMS
```hs
type City = String
type Distance = Int
type RoadMap = [(City, City, Distance)]

-- Get adjacent cities
adjacent :: RoadMap -> City -> [(City, Distance)]
adjacent roadMap city = 
    [(dest, dist) | (orig, dest, dist) <- roadMap, orig == city] ++
    [(orig, dist) | (orig, dest, dist) <- roadMap, dest == city]

-- Invert cities
inverse :: RoadMap -> RoadMap
inverse roadMap = [(y,x,d) | (x,y,d)<-roadMap]

newRoadMap :: RoadMap -> RoadMap
newRoadMap roadMap = [(x,z,d+d') | (x,y,d)<-roadMap, (y',z,d')<-roadMap, y'==y && x/=z]

-- DFS connectivity check
areConnected :: RoadMap -> City -> City -> Bool
areConnected roadMap start end = end `elem` dfs roadMap [] [start]
where
    dfs _ visited [] = visited
    dfs roadMap visited (top:stack)
        | top `elem` visited = dfs roadMap visited stack
        | otherwise = dfs roadMap (top:visited) (adjs ++ stack)
        where adjs = map fst $ adjacent roadMap top

-- Path distance
pathDistance :: RoadMap -> [City] -> Maybe Distance
pathDistance roadMap [] = Just 0
pathDistance roadMap (start:path) = aux start 0 path
where
    aux v acc [] = Just acc
    aux v acc (u:path) =
        case [d | (x,y,d) <- roadMap, x == v, y == u] of
            (d:_) -> aux u (acc+d) path
            [] -> Nothing

shortest :: RoadMap -> RoadMap
shortest [] = []
shortest ((x,y,d):roadMap) = (x,y,d'):shortest roadMap'
where
    d' = minimum (d : [d' | (a,b,d')<-roadMap, a==x && b==y])
    roadMap' = [(a,b,d) | (a,b,d)<-roadMap, a/=x || b/=y]
```
## 13. PROPOSITIONAL LOGIC
```hs
type VarName = Char

data Prop = Const Bool
        | Variable VarName
        | Not Prop
        | And Prop Prop
        | Imply Prop Prop
        deriving Show

valor :: Prop -> Bool
valor (Const b) = b
valor (Variable _) = False
valor (Not p) = not (valor p)
valor (And p q) = valor p && valor q
valor (Imply p q) = not (valor p) || valor q

rename :: [(VarName,VarName)] -> Prop -> Prop
rename _ (Const b) = Const b
rename ren (Variable x) = case lookup x ren of
                            Just y -> Variable y
                            Nothing -> Variable x
rename ren (Not p) = Not (rename ren p)
rename ren (And p q) = And (rename ren p) (rename ren q)
rename ren (Imply p q) = Imply (rename ren p) (rename ren q)

vars :: Prop -> [VarName]
vars (Const _) = []
vars (Variable x) = [x]
vars (Not p) = vars p
vars (And p q) = vars p ++ vars q
vars (Imply p q) = vars p ++ vars q

normalize :: Prop -> Prop
normalize p = rename (zip uniqueVars ['a'..]) p
where uniqueVars = nub (vars p)
```
## 14. ARITHMETIC EXPRESSIONS & EVALUATION
```hs
-- Expression data type for simple arithmetic
-- Val wraps an integer value
-- Add and Mul combine two expressions with + and *
data Expr = Val Int
        | Add Expr Expr
        | Mul Expr Expr
        deriving Show

-- Evaluate an Expr to an Int.
-- Evaluation is structural: recurse on sub‑expressions.
-- Example: eval (Add (Val 2) (Mul (Val 3) (Val 4))) == 14
eval :: Expr -> Int
eval (Val n)     = n
eval (Add e1 e2) = eval e1 + eval e2
eval (Mul e1 e2) = eval e1 * eval e2

-- Variants: to handle division or variables, extend Expr with more
-- constructors and lift eval into a Maybe type to handle errors.
```
## 15. LIST COMPREHENSIONS & SECTIONS
```hs
-- Translating list comprehensions into map/filter/concatMap:
-- A comprehension [f x y | x <- xs, y <- ys, p x y] is equivalent to:
-- concatMap (\x -> map (\y -> f x y) (filter (\y -> p x y) ys)) xs
--
-- Example: sum of squares of positive numbers
-- sum [x^2 | x <- xs, x > 0] = sum (map (^2) (filter (>0) xs))
--
-- Pairs of elements where the first is even:
-- [(x,y) | x <- xs, even x, y <- ys] =
--   concatMap (\x -> map (\y -> (x,y)) ys) (filter even xs)
--
-- Sections (partial application) shorthand:
-- (1+)    ≡ \x -> 1 + x        -- prefix constant
-- (+1)    ≡ \x -> x + 1        -- postfix constant
-- (*2)    ≡ \x -> x * 2
-- (/3)    ≡ \x -> x / 3
-- flip div 5 ≡ \x -> div x 5   -- sometimes flip is needed
--
-- You can use sections inside higher‑order functions:
-- map (*3) [1..5]       == [3,6,9,12,15]
-- filter (>=10) xs      -- keep elements ≥ 10
```
## 16. ADDITIONAL INFINITE SERIES & GENERATORS
```hs
-- cycle repeats a finite list indefinitely:
-- take 8 (cycle [1,2,3]) == [1,2,3,1,2,3,1,2]
--
-- iterate applies a function repeatedly:
-- iterate f x returns [x, f x, f (f x), ...]
-- Example: powers of two
powersOfTwo :: [Integer]
powersOfTwo = iterate (*2) 1
-- take 6 powersOfTwo == [1,2,4,8,16,32]

-- Alternating sign series using cycle and zipWith
-- Multiply each element by alternating +1 and -1
alternatingSum :: Num a => [a] -> a
alternatingSum xs = sum (zipWith (*) xs (cycle [1,-1]))

-- Example sequence: 1,-3,5,-7,9,...
oddAlternating :: [Integer]
oddAlternating = zipWith (*) [1,3..] (cycle [1,-1])

-- Using zipWith with iterate can define many recurrences
-- Example: geometric progression a_n = r * a_(n-1)
geo :: Num a => a -> a -> [a]
geo a0 r = iterate (*r) a0
```
## 17. COMMON EXERCISE FUNCTIONS & UTILITY DEFINITIONS
```hs
-- Remove the last element (init) without using the Prelude `init`.
-- Returns error on empty list.
initList :: [a] -> [a]
initList [] = error "empty"
initList [_] = []
initList (x:xs) = x : initList xs

-- Drop the first and last elements, producing the "middle" of a list.
middle :: [a] -> [a]
middle = initList . tail

-- Exclusive-or on booleans
xor :: Bool -> Bool -> Bool
xor p q = (p || q) && not (p && q)

-- Safe tail: returns empty list when given an empty list.
safetail :: [a] -> [a]
safetail [] = []
safetail (_:xs) = xs

-- Check if a list is short (has fewer than 3 elements).
short :: [a] -> Bool
short xs = length xs < 3

-- Median of three values by sorting them.
medianOfThree :: Ord a => a -> a -> a -> a
medianOfThree a b c = sortByCond [a,b,c] (<=) !! 1

-- Median of a list.  For odd length, take middle element; for even, average the two middle elements.
median :: (Fractional a, Ord a) => [a] -> a
median xs =
    let sorted = sortByCond xs (<=)
        n = length xs
    in if odd n
        then sorted !! (n `div` 2)
        else let i = n `div` 2
                a1 = sorted !! (i - 1)
                a2 = sorted !! i
            in (a1 + a2) / 2

isPrime :: Integral a => a -> Bool
isPrime n
    | n <= 1    = False
    | otherwise = null [x | x <- [2..floor (sqrt (fromIntegral n))], n `mod` x == 0]
```
## 18. BINARY CONVERSIONS, GROUPING & HAMMING NUMBERS
```hs
-- Convert a non‑negative integer to a list of bits (most significant bit first).
toBits :: Integral a => a -> [Int]
toBits 0 = [0]
toBits n = reverse (helper n)
where helper 0 = []
        helper x = let (q,r) = x `divMod` 2 in r : helper q

-- Convert a list of bits back to an integer.  Assumes most significant bit first.
fromBits :: [Int] -> Int
fromBits = foldl (\acc bit -> acc * 2 + bit) 0

-- Intercalate a separator between sublists.
intercalate :: a -> [[a]] -> [a]
intercalate _ [] = []
intercalate _ [xs] = xs
intercalate sep (xs:xss) = xs ++ [sep] ++ intercalate sep xss

-- Generate all permutations of a list by recursively interleaving.
permutations :: [a] -> [[a]]
permutations [] = [[]]
permutations (x:xs) = concatMap (interleave x) (permutations xs)
where interleave x [] = [[x]]
        interleave x (y:ys) = (x:y:ys) : map (y:) (interleave x ys)

-- Infinite list of Hamming numbers (numbers whose prime factors are 2,3 or 5).
hamming :: [Integer]
hamming = 1 : merge (map (*2) hamming) (merge (map (*3) hamming) (map (*5) hamming))
where merge (x:xs) (y:ys)
            | x < y     = x : merge xs (y:ys)
            | x > y     = y : merge (x:xs) ys
            | otherwise = x : merge xs ys
```
## 19. CUSTOM DATA TYPES & LOGIC UTILITIES
```hs
-- Extended propositional logic with disjunction.
data Prop2 = Const2 Bool
        | Var VarName
        | Not2 Prop2
        | And2 Prop2 Prop2
        | Or2  Prop2 Prop2
        | Imply2 Prop2 Prop2
        deriving Show

-- Environments assign boolean values to variables.
type Env = [(VarName, Bool)]

-- Evaluate a proposition under an environment.
aval :: Env -> Prop2 -> Bool
aval _   (Const2 b)    = b
aval env (Var x)       = case lookup x env of
                        Just v  -> v
                        Nothing -> False
aval env (Not2 p)      = not (aval env p)
aval env (And2 p q)    = aval env p && aval env q
aval env (Or2  p q)    = aval env p || aval env q
aval env (Imply2 p q)  = not (aval env p) || aval env q

-- Collect variables in a proposition (without duplicates).
listVar :: Prop2 -> [VarName]
listVar (Const2 _)    = []
listVar (Var x)       = [x]
listVar (Not2 p)      = listVar p
listVar (And2 p q)    = nub (listVar p ++ listVar q)
listVar (Or2  p q)    = nub (listVar p ++ listVar q)
listVar (Imply2 p q)  = nub (listVar p ++ listVar q)

-- Generate all boolean combinations of length n.
bools :: Int -> [[Bool]]
bools 0 = [[]]
bools n = [b:bs | b <- [False, True], bs <- bools (n-1)]

-- Generate all environments for a list of variables.
envs :: [VarName] -> [Env]
envs vars = [ zip vars vals | vals <- bools (length vars) ]

-- Truth table: list of (environment, evaluation) pairs.
tablen :: Prop2 -> [(Env, Bool)]
tablen p = [ (env, aval env p) | env <- envs vs ]
where vs = listVar p

-- All satisfying environments (models) of a proposition.
satisfies :: Prop2 -> [Env]
satisfies p = [ env | env <- envs vs, aval env p ]
where vs = listVar p
```
# PROLOG
## 1. BASIC RECURSION PATTERNS
```prolog
% Factorial
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

% Tail-recursive factorial
factorial_tr(N, F) :- factorial_acc(N, 1, F).
factorial_acc(0, Acc, Acc).
factorial_acc(N, Acc, F) :- 
    N > 0, Acc1 is Acc * N, N1 is N - 1, factorial_acc(N1, Acc1, F).

% Fibonacci
fib(0, 0).
fib(1, 1).
fib(N, F) :- 
    N > 1, N1 is N - 1, N2 is N - 2, 
    fib(N1, F1), fib(N2, F2), F is F1 + F2.

% Power
pow(_, 0, 1).
pow(X, 1, X).
pow(X, Y, P) :- Y > 1, Y1 is Y - 1, pow(X, Y1, P1), P is X * P1.

% Sum from 1 to N
sum_rec(0, 0).
sum_rec(N, S) :- N > 0, N1 is N - 1, sum_rec(N1, S1), S is N + S1.
```
## 2. LIST OPERATIONS
```prolog
% List size
list_size([], 0).
list_size([_|T], Size) :- list_size(T, SizeT), Size is SizeT + 1.

% List sum
list_sum([], 0).
list_sum([H|T], Sum) :- list_sum(T, SumT), Sum is SumT + H.

% List product
list_prod([], 1).
list_prod([H|T], Prod) :- list_prod(T, ProdT), Prod is ProdT * H.

% Reverse
invert([], []).
invert([H|T], R) :- invert(T, RT), append(RT, [H], R).

% Better reverse (tail-recursive)
reverse_tr(L, R) :- reverse_acc(L, [], R).
reverse_acc([], Acc, Acc).
reverse_acc([H|T], Acc, R) :- reverse_acc(T, [H|Acc], R).

% Delete all occurrences
del_all(_, [], []).
del_all(X, [X|T], R) :- del_all(X, T, R).
del_all(X, [H|T], [H|R]) :- X \= H, del_all(X, T, R).

% Count occurrences
count(_, [], 0).
count(X, [X|T], N) :- count(X, T, N1), N is N1 + 1.
count(X, [H|T], N) :- X \= H, count(X, T, N).

% Remove duplicates
del_dups([], []).
del_dups([H|T], [H|R]) :- \+ member(H, T), del_dups(T, R).
del_dups([H|T], R) :- member(H, T), del_dups(T, R).

% Insert at position
insert_elem(1, List, Elem, [Elem|List]).
insert_elem(I, [X|Xs], Elem, [X|R]) :- 
    I > 1, I1 is I - 1, insert_elem(I1, Xs, Elem, R).

% Delete at position
delete_elem(1, [X|Xs], X, Xs).
delete_elem(I, [X|Xs], Elem, [X|R]) :- 
    I > 1, I1 is I - 1, delete_elem(I1, Xs, Elem, R).

% Replace at position
replace([Old|Xs], 1, Old, New, [New|Xs]).
replace([X|Xs], I, Old, New, [X|R]) :- 
    I > 1, I1 is I - 1, replace(Xs, I1, Old, New, R).

% Maximum element
max_list([X], X).
max_list([H|T], Max) :-
    max_list(T, M1),
    (H >= M1 -> Max = H ; Max = M1).
```
## 3. APPEND-BASED PATTERNS
```prolog
% Append definition
list_append([], L, L).
list_append([H|T], L, [H|R]) :- list_append(T, L, R).

% Member using append
list_member(X, L) :- append(_, [X|_], L).

% Last element
list_last(L, X) :- append(_, [X], L).

% Nth element
list_nth(N, L, X) :- length(Prefix, N1), N is N1 + 1, append(Prefix, [X|_], L).

% Before predicate
before(F, S, L) :- append(_, [F|L2], L), append(_, [S|_], L2).

% Delete one occurrence
list_del(L, X, R) :- append(Front, [X|Back], L), append(Front, Back, R).
```
## 4. HIGHER-ORDER PREDICATES
```prolog
% Map (applies predicate to all elements)
map(_, [], []).
map(P, [H|T], [R|RT]) :-
    G =.. [P, H, R],
    call(G),
    map(P, T, RT).

% Filter
filter(_, [], []).
filter(P, [H|T], [H|R]) :-
    G =.. [P, H],
    call(G),
    filter(P, T, R).
filter(P, [_|T], R) :- filter(P, T, R).

% Fold (reduce)
fold(_, Acc, [], Acc).
fold(P, Acc0, [H|T], AccF) :-
    G =.. [P, Acc0, H, Acc1],
    call(G),
    fold(P, Acc1, T, AccF).
```
## 5. FINDALL, BAGOF, SETOF
```prolog
% findall(X, Goal, List) - always succeeds, returns [] if no solutions
% bagof(X, Goal, List) - fails if no solutions, may return multiple lists
% setof(X, Goal, List) - sorted & unique, fails if no solutions

% Common patterns:
% findall(X, Goal, L).                      % collect all solutions
% findall(A-B, Goal, L).                    % collect pairs
% findall(X, (G1, findall(Y, G2, Ys)), L).  % nested collection
% setof(X, Y^Goal, L).                      % existential quantification

% Example: Count dishes with ingredient
count_dishes_with_ingredient(Ingredient, N) :-
    findall(Dish, (dish(Dish, _, Ingredients), member(Ingredient-_, Ingredients)), Dishes),
    length(Dishes, N).
```
## 6. GRAPH SEARCH (DFS/BFS)
```prolog
% DFS (with cycle detection)
dfs(Goal, Goal, _, []).
dfs(Start, Goal, Visited, [Edge|Path]) :-
    edge(Start, Next, Edge),
    \+ member(Next, Visited),
    dfs(Next, Goal, [Next|Visited], Path).

% BFS (using queue)
bfs(Start, Goal, Path) :- bfs_queue([[Start]], Goal, Path).

bfs_queue([[Goal|Rest]|_], Goal, Path) :- reverse([Goal|Rest], Path).
bfs_queue([Path|Others], Goal, Final) :-
    Path = [Node|_],
    findall([Next|Path],
            (edge(Node, Next, _), \+ member(Next, Path)),
            Expansions),
    append(Others, Expansions, NewQueue),
    bfs_queue(NewQueue, Goal, Final).

% Connectivity check
connects(X, Y, Path) :- connects(X, [X], Y, Path).
connects(X, Path, X, RevPath) :- reverse(Path, RevPath).
connects(X, Visited, Y, Path) :-
    edge(X, Z, _),
    \+ member(Z, Visited),
    connects(Z, [Z|Visited], Y, Path).
```
## 7. DYNAMIC PREDICATES & ASSERT/RETRACT
```prolog
% Memoized Fibonacci
:- dynamic fib_memo/2.

fib_memo(0, 0).
fib_memo(1, 1).
fib_memo(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib_memo(N1, F1),
    fib_memo(N2, F2),
    F is F1 + F2,
    asserta(fib_memo(N, F)).

% Update ingredient cost
update_unit_cost(Ingredient, NewCost) :-
    retractall(ingredient(Ingredient, _)),
    asserta(ingredient(Ingredient, NewCost)).
```
## 8. TREES IN PROLOG
```prolog
% Tree structure: nil or t(Value, Left, Right)

% Size
tree_size(nil, 0).
tree_size(t(_, L, R), N) :- 
    tree_size(L, NL), tree_size(R, NR), N is NL + NR + 1.

% Height
tree_height(nil, 0).
tree_height(t(_, L, R), H) :-
    tree_height(L, HL), tree_height(R, HR),
    H is max(HL, HR) + 1.

% Leaves count
tree_leaves(nil, 0).
tree_leaves(t(_, nil, nil), 1).
tree_leaves(t(_, L, R), N) :-
    tree_leaves(L, NL), tree_leaves(R, NR),
    N is NL + NR.

% Tree map
tree_map(_, nil, nil).
tree_map(P, t(X, L, R), t(Y, NL, NR)) :-
    G =.. [P, X, Y], call(G),
    tree_map(P, L, NL), tree_map(P, R, NR).
```
## 9. META-PROGRAMMING
```prolog
% Univ operator (=..)
% Converts between term and list representation
% T =.. [Functor|Args]

% Examples:
% foo(a, b, c) =.. [foo, a, b, c]
% T =.. [+, 2, 3] gives T = 2 + 3

% Get functor and arity
my_functor(T, F, N) :- T =.. [F|Args], length(Args, N).

% Get nth argument
my_arg(I, T, A) :- T =.. [_|Args], nth1(I, Args, A).

% Test if second argument is uninstantiated
test_second_uninstantiated(T) :-
    T =.. [_, _, X|_],
    var(X).
```
## 10. CUT & NEGATION
```prolog
% Green cut (doesn't change semantics, only efficiency)
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).

% Red cut (changes semantics)
min_list([X], X) :- !.
min_list([H|T], M) :- min_list(T, M1), M is min(H, M1).

% Negation as failure
\+ Goal   % Goal fails

% Not member
not_member(_, []).
not_member(X, [H|T]) :- X \= H, not_member(X, T).
```
## 11. OPERATORS
```prolog
% Define custom operators:
% :- op(Precedence, Type, Name).

% Types:
%   xfx - infix, non-associative  (e.g., =, <)
%   yfx - infix, left-associative (e.g., +, -)
%   xfy - infix, right-associative (e.g., ^)
%   fx  - prefix  (e.g., -)
%   fy  - prefix  (e.g., \+)
%   xf  - postfix
%   yf  - postfix

% Example:
:- op(700, xfx, has).
:- op(600, xf, pages).
% Allows: 'The Firm' has 432 pages.
```
## 12. COMMON EXAM PATTERNS
```prolog
siblings(X, Y) :- 
    parent(P, X), parent(P, Y), X \= Y.

grandparent(X, Y) :- 
    parent(X, Z), parent(Z, Y).

% Pascal's triangle
pascal(1, [[1]]).
pascal(N, Lines) :-
    N > 1, N1 is N - 1,
    pascal(N1, Prev),
    append(_, [LastLine], Prev),
    next_pascal_line(LastLine, New),
    append(Prev, [New], Lines).

next_pascal_line([X|Xs], [X|Ys]) :- next_aux(Xs, X, Ys).
next_aux([], Last, [Last]).
next_aux([X|Xs], Prev, [Sum|Ys]) :-
    Sum is Prev + X, next_aux(Xs, X, Ys).

% Unifiable elements
unifiable([], _, []).
unifiable([H|T], Term, [H|R]) :-
    \+ \+ (H = Term),  % Test unification without binding
    unifiable(T, Term, R).
unifiable([_|T], Term, R) :- unifiable(T, Term, R).
```
## 13. PERMUTATIONS & COLOR ANALYSIS
```prolog
% Check if two lists represent the same multiset (permutation).
% This uses msort/2 to sort both lists and then compares.
is_permutation(L1, L2) :-
    msort(L1, S), msort(L2, S).

% Alternative using built‑in permutation/2 (generates permutations).
same_elements(L1, L2) :-
    permutation(L1, L2).  % succeeds if L2 is a permutation of L1

% Count occurrences of an element in a list (counts duplicates).
count_occ(_, [], 0).
count_occ(X, [X|T], N) :-
    count_occ(X, T, N1),
    N is N1 + 1.
count_occ(X, [_|T], N) :-
    count_occ(X, T, N).

% Example: count how many times a colour appears in a 3‑colour tuple.
% count_colour(red,(C1,C2,C3),N) unifies N with the number of reds.
count_colour(C, (C1,C2,C3), N) :-
    findall(1, (nth1(_, [C1,C2,C3], C)), Ones),
    length(Ones, N).

% Determining birds with the same colour set:
% Suppose we have facts colours(birdName,[ListOfColours]).
% To check if Bird1 and Bird2 have the same colours (ignoring order):
same_colours(Bird1, Bird2) :-
    colours(Bird1, C1),
    colours(Bird2, C2),
    is_permutation(C1, C2).
```
## 14. ADVANCED OPERATORS & CUTS
```prolog
% Custom operators allow more readable domain‑specific syntax.  For example:
% define a cost relation "costs" and a currency "euros".
:- op(500, xfx, costs).
:- op(400, xf, euros).

% Now you can write a fact like:
%   budget costs 20 euros.
% which Prolog interprets as costs(budget,20) combined with euros(20).

% CUT CLASSIFICATION:
% A cut (!/0) prunes choice points.  A green cut is used only to remove
% redundant searches and does not affect the set of solutions.  A red cut
% changes the semantics by discarding alternative solutions.
%
% Example of a green cut:
%   max(X,Y,X) :- X >= Y, !.
%   max(_,Y,Y).
% The cut prevents unnecessary testing of the second clause once the first
% succeeds.
%
% Example of a red cut:
%   grade(Score, pass) :- Score >= 10, !.
%   grade(_, fail).
% The cut here forces any score >= 10 to be classified as pass and suppresses
% other possible clauses, potentially excluding valid alternatives.
```
## 15. LIST & UTILITY PREDICATES
```prolog
% Delete first occurrence of an element from a list.
del_one(_, [], []) :- !.
del_one(X, [X|T], T) :- !.
del_one(X, [H|T], [H|R]) :- del_one(X, T, R).

% Replicate an element N times.
replicate_elem(0, _, []).
replicate_elem(N, X, [X|R]) :- N > 0, N1 is N - 1, replicate_elem(N1, X, R).

% Intersperse a separator between list elements.
intersperse_elem(_, [], []).
intersperse_elem(_, [X], [X]).
intersperse_elem(Sep, [H|T], [H,Sep|R]) :-
    intersperse_elem(Sep, T, R).

% Extract a slice from position I to J (1‑based, inclusive).
slice(L, I, J, S) :-
    I > 0, J >= I,
    length(Prefix, I1), I1 is I - 1,
    append(Prefix, Rest, L),
    K is J - I + 1,
    length(S, K),
    append(S, _, Rest).

% Rotate a list left by one position.
rotate_left([], []).
rotate_left([H|T], R) :- append(T, [H], R).

% Generate list of integers from A to B.
from_to(A, B, []) :- A > B.
from_to(A, B, [A|R]) :-
    A =< B,
    A1 is A + 1,
    from_to(A1, B, R).

% Generate list from A to B with step.
from_to_step(A, B, _, []) :-
    (A > B).
from_to_step(A, B, Step, [A|R]) :-
    Step > 0, A =< B,
    A1 is A + Step,
    from_to_step(A1, B, Step, R).
from_to_step(A, B, Step, [A|R]) :-
    Step < 0, A >= B,
    A1 is A + Step,
    from_to_step(A1, B, Step, R).

% Simple primality test and list of primes up to N.
prime(2).
prime(P) :-
    integer(P), P > 2,
    P mod 2 =\= 0,
    \+ has_factor(P, 3).
has_factor(N,F) :-
    F*F =< N,
    (N mod F =:= 0 ; F2 is F + 2, has_factor(N,F2)).

primes_up_to(N, Ps) :- findall(P, (between(2,N,P), prime(P)), Ps).

% Generate Fibonacci numbers up to N
fib_upto(N, Fibs) :- fib_seq(0, 1, N, Fibs).
fib_seq(A, _, N, []) :- A > N, !.
fib_seq(A, B, N, [A|Rest]) :-
    A =< N,
    C is A + B,
    fib_seq(B, C, N, Rest).
```
## 16. INPUT/OUTPUT & FORMATTED PRINTING
```prolog
% Print N copies of a character or atom.
print_n(0, _) :- !.
print_n(N, X) :-
    N > 0,
    write(X),
    N1 is N - 1,
    print_n(N1, X).

% Print an ASCII Christmas tree of height N.
oh_christmas_tree(N) :- oh_tree_rows(1, N).
oh_tree_rows(Level, N) :-
    Level =< N,
    Spaces is N - Level,
    Stars is 2 * Level - 1,
    print_n(Spaces, ' '), print_n(Stars, '*'), nl,
    Level1 is Level + 1,
    oh_tree_rows(Level1, N).
oh_tree_rows(_, _).

% Print a list on one line, space‑separated.
print_full_list([]) :- nl.
print_full_list([H|T]) :-
    write(H),
    (T = [] -> nl ; write(' '), print_full_list(T)).

% Print at most the first three elements of a list, adding '...' if more follow.
print_list(L) :-
    ( length(L, Len), Len =< 3 ->
        print_full_list(L)
    ; L = [A,B,C|_] ->
        write(A), write(' '), write(B), write(' '), write(C), write(' ...'), nl
    ).

% Print a matrix (list of lists) line by line.
print_matrix([]).
print_matrix([Row|Rest]) :- print_full_list(Row), print_matrix(Rest).
```
## 17. HIGHER‑ORDER & META‑PROGRAMMING PATTERNS
```prolog
% separate(Pred, List, Yes, No) splits List into elements for which Pred succeeds and those it fails.
separate(_, [], [], []).
separate(P, [H|T], [H|Yes], No) :-
    G =.. [P, H],
    call(G),
    separate(P, T, Yes, No).
separate(P, [H|T], Yes, [H|No]) :-
    G =.. [P, H],
    \+ call(G),
    separate(P, T, Yes, No).

% take_while(Pred, List, Prefix) collects elements until Pred fails.
take_while(_, [], []).
take_while(P, [H|T], [H|R]) :-
    G =.. [P, H],
    call(G),
    take_while(P, T, R).
take_while(P, [H|_], []) :-
    G =.. [P, H],
    \+ call(G).

% ask_execute reads a goal from the user and executes it.
ask_execute :-
    write('Enter a goal (ending with a period): '), nl,
    read(Goal),
    ( call(Goal) -> writeln('Yes') ; writeln('No') ).

% Meta‑predicates using =.. (univ).
my_functor(Term, Functor, Arity) :-
    Term =.. [Functor|Args],
    length(Args, Arity).

my_arg(I, Term, Arg) :-
    Term =.. [_|Args],
    nth1(I, Args, Arg).

% my_univ relates a term and its list representation.
my_univ(Term, List) :- Term =.. List.

% tree_value_at_level(Tree, Level, Values) collects values at a given depth.
tree_value_at_level(nil, _, []).
tree_value_at_level(t(X, _, _), 0, [X]).
tree_value_at_level(t(_, L, R), N, Values) :-
    N > 0,
    N1 is N - 1,
    tree_value_at_level(L, N1, VL),
    tree_value_at_level(R, N1, VR),
    append(VL, VR, Values).
```
# EXAM THEORY QUICK REFERENCE
## HASKELL TYPE ERRORS:
```hs
✗ 1 + False          -- Can't add Int and Bool
✗ 'a' + 'b'          -- Can't add Chars
✗ 'a' ++ 'b'         -- ++ needs [Char], not Char
✗ 1+2 == "3"         -- Can't compare Int and String
✗ 'a' < "ab"         -- Can't compare Char and String
✗ (1 <= 2) <= 3      -- Can't compare Bool and Int
✗ head (1,2)         -- head needs list, not tuple
```
## TYPE vs DATA:
- Only "data" allows recursive definitions
- Only "data" creates new pattern matching constructors
- Both allow type variables
## EVALUATION ORDER:
- Lazy evaluation enables infinite data structures
- `takeWhile (<1000) [1..]` terminates
- `length (filter (<1000) fibs)` does NOT terminate (filter needs full evaluation)
## FOLD DIRECTION:
- `foldr (+) 7 [1,2,3]` = 1 + (2 + (3 + 7)) = 13
- `foldl (/) 200 [1,2,4]` = ((200 / 1) / 2) / 4 = 25.0
## TYPECLASSES:
- Every Ord instance is also an Eq instance
- (/) is in Fractional, not Integral
- Int is instance of Eq, Ord, Num
## MONADS:
- Maybe, State, IO are all monads
- IO () means IO action returning unit
## PROLOG FACTS:
- findall always succeeds (returns [] if none)
- bagof/setof fail if no solutions
- setof returns sorted, unique results
- Use `Y^` to existentially quantify variables: `setof(X, Y^goal(X,Y), L)`
## CUT COLORS:
- Green cut: doesn't change semantics (only efficiency)
- Red cut: changes semantics (removes choice points that affect correctness)
## NEGATION:
- `\+` Goal succeeds if Goal fails (negation as failure)
- Not the same as logical negation!
# USEFUL PATTERNS
## HASKELL COMMON IDIOMS:
- `filter p . map f` == `map f . filter (p . f)` [when p doesn't depend on f]
- `length . filter p` == length of elements satisfying p
- `foldr (++) []` == concat
- `map f` == `foldr ((:) . f) []`
- `filter p` == `foldr (\x acc -> if p x then x:acc else acc) []`
## PROLOG COMMON IDIOMS:
```prolog
member(X, L) :- % append(_, [X|_], L).
last(L, X) :- % append(_, [X], L).
prefix(P, L) :- % append(P, _, L).
suffix(S, L) :- % append(_, S, L).
reverse(L, R) :- % reverse(L, [], R).
sublist(Sub, L) :- % append(_, Temp, L), append(Sub, _, Temp).
```
## ARITHMETIC EVALUATION:
```prolog
- is: X is Expr           % Evaluates Expr and unifies with X
Example: X is 2 + 3       % X = 5
Example: Y is 2 * (3 + 4) % Y = 14
% - Expr is evaluated: +, -, *, /, //, mod, **, abs, max, min, sqrt, etc.
% - Variables on right side must be instantiated!
✗ X is Y + 1 (if Y unbound) %% ERROR
✓ Y = 5, X is Y + 1         %% X = 6
```
## ARITHMETIC COMPARISON (evaluates both sides):
```prolog
=:=: X =:= Y                % Arithmetic equality (evaluates expressions)
% Example: 2 + 3 =:= 5      % true
% Example: 2 + 3 =:= 6      % false
=\=: X =\= Y                % Arithmetic inequality
% Example: 2 + 3 =\= 6      % true
<: X < Y                    % Less than
>: X > Y                    % Greater than
=<: X =< Y                  % Less than or equal (NOTE: =< not <=)
>=: X >= Y                  % Greater than or equal
```
## TERM COMPARISON (no evaluation):
```prolog
==: X == Y                  % Strict equality (no unification)
% Example: X == Y           % true only if X and Y already identical
% Example: 1 + 2 == 3       % false (terms differ structurally)
% Example: 1 + 2 == 1 + 2   % true (same structure)
\==: X \== Y                % Strict inequality
% Example: f(X) \== f(Y)    % true if X and Y are different variables
@<: X @< Y                  % Term less than (standard order)
@>: X @> Y                  % Term greater than
@=<: X @=< Y                % Term less or equal
@>=: X @>= Y                % Term greater or equal
```
## UNIFICATION:
```prolog
=: X = Y                    % Unification (tries to make X and Y identical)
% Example: X = 5            % binds X to 5
% Example: f(X) = f(3)      % binds X to 3
% Example: [H|T] = [1,2,3]  % H = 1, T = [2,3]
% Example: 1 + 2 = 3        % false (doesn't evaluate, just structural match)
\=: X \= Y                  % Not unifiable
% Example: f(a) \= f(b)     % true
% Example: X \= 5           % constrains X to not be 5
```
## STANDARD TERM ORDER (for @<, @>, etc.):
Variables < Numbers < Atoms < Compound Terms
- Variables ordered alphabetically by name
- Numbers ordered by value
- Atoms ordered alphabetically
- Compound terms ordered by: arity, then functor name, then arguments
## COMMON PITFALLS:
```prolog
✗ X = Y + 1                 % Does NOT evaluate! Just unifies X with term +(Y,1)
✓ X is Y + 1                % Evaluates Y + 1 and binds result to X
✗ 2 + 3 == 5                % false (compares terms, not values)
✓ 2 + 3 =:= 5               % true (evaluates both sides)
✗ X =< 10                   % ERROR if X unbound (use for guards after instantiation)
✓ between(1, 10, X)         % generates X from 1 to 10
```
## TYPE CHECKING PREDICATES:
```prolog
- var(X)                    % true if X is uninstantiated variable
- nonvar(X)                 % true if X is instantiated
- atom(X)                   % true if X is an atom
- number(X)                 % true if X is a number
- integer(X)                % true if X is an integer
- float(X)                  % true if X is a float
- compound(X)               % true if X is a compound term
- is_list(X)                % true if X is a proper list
- atomic(X)                 % true if X is atom or number
- ground(X)                 % true if X contains no variables
```
## EXAMPLES:
```prolog
% Arithmetic
?- X is 10 / 3.             % X = 3.333...
?- X is 10 // 3.            % X = 3 (integer division)
?- X is 10 mod 3.           % X = 1
?- X is 2 ** 10.            % X = 1024

% Comparison
?- 5 =:= 2 + 3.             % true (arithmetic equality)
?- 5 == 2 + 3.              % false (term comparison)
?- 5 = 2 + 3.               % false (unification fails)
?- X = 2 + 3.               % X = 2 + 3 (creates term, doesn't eval)
?- X is 2 + 3, X =:= 5.     % true
```
# 2025/26
## Haskell
```hs
type Node = String   -- some city
type Dist = Int      -- some distance
type Edges = [(Node,Node,Dist)] -- directed connections
For example, the following edges represent some connections in Portugal (the distances are in kilometers).

portugal :: Edges
portugal = [ ("Porto", "Aveiro", 76)
           , ("Aveiro", "Coimbra", 63)
           , ("Aveiro", "Leiria", 117)
           , ("Coimbra", "Leiria", 76)
           , ("Leiria", "Santarem", 83)
           , ("Santarem", "Lisboa", 82)
           ]

inverse :: Edges -> Edges
inverse edges = [(y,x,d) | (x,y,d)<-edges]

newEdges :: Edges -> Edges
newEdges edges = [(x,z,d+d') | (x,y,d)<-edges, (y',z,d')<-edges, y'==y && x/=z]

pathDistance :: Edges -> [Node] -> Maybe Dist
pathDistance edges [] = Just 0
pathDistance edges (start:path) = aux start 0 path
  where
    aux v acc [] = Just acc
    aux v acc (u:path) =
      case [d | (x,y,d)<-edges, x==v && y==u] of
        (d:_) -> aux u (acc+d) path
        [] -> Nothing

shortest :: Edges -> Edges
shortest [] = []
shortest ((x,y,d):edges) = (x,y,d'):shortest edges'
  where
    d' = minimum (d : [d' | (a,b,d')<-edges, a==x && b==y])
    edges' = [(a,b,d) | (a,b,d)<-edges, a/=x || b/=y]

nodes :: Edges -> [Node]
nodes edges = nub ([x | (x,_,_) <- edges] ++ [y | (_,y,_) <- edges])
  where nub [] = []
        nub (x:xs) = x : nub (filter (/= x) xs)

outgoing :: Node -> Edges -> Edges
outgoing node edges = [(x,y,d) | (x,y,d) <- edges, x == node]

incoming :: Node -> Edges -> Edges
incoming node edges = [(x,y,d) | (x,y,d) <- edges, y == node]

neighbors :: Node -> Edges -> [(Node, Dist)]
neighbors node edges = [(y,d) | (x,y,d) <- edges, x == node]

connected :: Node -> Node -> Edges -> Bool
connected x y edges = any (\(a,b,_) -> a == x && b == y) edges

directDistance :: Node -> Node -> Edges -> Maybe Dist
directDistance x y edges = case [d | (a,b,d) <- edges, a == x, b == y] of
  (d:_) -> Just d
  [] -> Nothing

undirected :: Edges -> Edges
undirected edges = edges ++ inverse edges

filterByDistance :: (Dist -> Bool) -> Edges -> Edges
filterByDistance pred edges = [(x,y,d) | (x,y,d) <- edges, pred d]

shorterThan, longerThan :: Dist -> Edges -> Edges
shorterThan maxDist = filterByDistance (< maxDist)
longerThan minDist = filterByDistance (> minDist)

totalDistance :: Edges -> Dist
totalDistance edges = sum [d | (_,_,d) <- edges]

averageDistance :: Edges -> Double
averageDistance edges = fromIntegral (totalDistance edges) / fromIntegral (length edges)

longestEdge :: Edges -> Maybe (Node, Node, Dist)
longestEdge [] = Nothing
longestEdge edges = Just (maxBy (\(_,_,d1) (_,_,d2) -> compare d1 d2) edges)
  where maxBy _ [x] = x
        maxBy cmp (x:xs) = let y = maxBy cmp xs in if cmp x y == GT then x else y

shortestEdge :: Edges -> Maybe (Node, Node, Dist)
shortestEdge [] = Nothing
shortestEdge edges = Just (minBy (\(_,_,d1) (_,_,d2) -> compare d1 d2) edges)
  where minBy _ [x] = x
        minBy cmp (x:xs) = let y = minBy cmp xs in if cmp x y == LT then x else y

validPath :: Edges -> [Node] -> Bool
validPath _ [] = True
validPath _ [_] = True
validPath edges (x:y:rest) = connected x y edges && validPath edges (y:rest)

destinations :: Edges -> [Node]
destinations edges = [n | n <- nodes edges, null (outgoing n edges)]

sources :: Edges -> [Node]
sources edges = [n | n <- nodes edges, null (incoming n edges)]

outDegree :: Node -> Edges -> Int
outDegree node edges = length (outgoing node edges)

inDegree :: Node -> Edges -> Int
inDegree node edges = length (incoming node edges)

pathsOfLength :: Int -> Node -> Node -> Edges -> [[Node]]
pathsOfLength 0 start end _ = if start == end then [[start]] else []
pathsOfLength n start end edges
  | n < 0 = []
  | otherwise = [[start] ++ path | (next, _) <- neighbors start edges, 
                                     path <- pathsOfLength (n-1) next end edges]
```

## Prolog
```prolog
bird(robinho,    robin,   male,   [red, brown, white]).
bird(robina,     robin,   female, [brown, red, white]).
bird(ferrugem,   robin,   male,   [brown, gray]).

bird(arcoiris,   parrot,  male,   [red, blue, green, yellow]).
bird(verdeja,    parrot,  female, [green, yellow]).

bird(minerva,    owl,     female, [brown, white]).
bird(noctis,     owl,     male,   [gray, white]).
bird(sabia,      owl,     female, [beige, brown]).

male(Name):-
    bird(Name, _, male, _).

has_more_color_of(N, C1, C2):-
    bird(N, _, _, Cs),
    append(_, [C1|T], Cs),
    append(_, [C2|_], T).

most_colorful(Sp, N, NC):-
    bird(N, Sp, _, Cs),
    length(Cs, NC),
    \+((
        bird(_, Sp, _, Cs1),
        length(Cs1, NC1),
        NC1 > NC
    )).

unique_colors(Sp, L):-
    unique_colors_aux(Sp, [], L).

unique_colors_aux(Sp, Acc, Sol):-
    bird(_, Sp, _, Cs),
    member(C, Cs),
    \+ member(C, Acc),
    !,
    unique_colors_aux(Sp, [C|Acc], Sol).
unique_colors_aux(_, Acc, Acc).

is_color_permutation(N1, N2) :-
    bird(N1, _, _, Cs1),
    bird(N2, _, _, Cs2),
    N1 \= N2,
    sort(Cs1, Sorted),
    sort(Cs2, Sorted).

dif_n_colors(Sp, D):-
    findall(N, ( bird(_, Sp, _, Cs), length(Cs, N) ), L),
    sort(L, Sorted),
    last(Sorted, Max),
    Sorted = [Min|_],
    D is Max - Min.

most_common_color_per_species(Sp, Color):-
    bagof(C, (N,G,Cs)^( bird(N, Sp, G, Cs), member(C, Cs) ), Colors),
    setof(N-C, (Rest,All,LRest)^( member(C, Colors), delete(Colors, C, Rest), length(Colors, All), length(Rest, LRest), N is All-LRest ), Counts),
    last(Counts, MaxN-_),
	member(MaxN-Color, Counts).

colorful_routes(Ni, Nf, L):-
    L = [_, _, _, _, _ | _],
    dfs([Ni], Nf, L).

dfs([Nf|_], Nf, [Nf]).
dfs([Na|Acc], Nf, [Na|Path]):-
    bird(Na, _, _, [Ca|_]),
    bird(Nb, _, _, [Cb|_]),
    Ca \= Cb,
    \+ member(Nb, Acc),
    dfs([Nb,Na|Acc], Nf, Path).

birds_of_species(Species, Birds):-
    findall(Name, bird(Name, Species, _, _), Birds).

birds_by_gender(Gender, Birds):-
    findall(Name, bird(Name, _, Gender, _), Birds).

female(Name):-
    bird(Name, _, female, _).

count_species(Species, Count):-
    findall(Name, bird(Name, Species, _, _), Birds),
    length(Birds, Count).

count_gender(Gender, Count):-
    findall(Name, bird(Name, _, Gender, _), Birds),
    length(Birds, Count).

has_color(Name, Color):-
    bird(Name, _, _, Colors),
    member(Color, Colors).

birds_with_color(Color, Birds):-
    findall(Name, (bird(Name, _, _, Colors), member(Color, Colors)), Birds).

color_count(Name, Count):-
    bird(Name, _, _, Colors),
    length(Colors, Count).

least_colorful(Species, Name, Count):-
    bird(Name, Species, _, Colors),
    length(Colors, Count),
    \+ (
        bird(_, Species, _, OtherColors),
        length(OtherColors, OtherCount),
        OtherCount < Count
    ).

all_species(Species):-
    setof(S, (N,G,C)^bird(N, S, G, C), Species).

all_colors(Colors):-
    findall(C, (bird(_, _, _, Cs), member(C, Cs)), AllColors),
    sort(AllColors, Colors).

share_color(Name1, Name2):-
    bird(Name1, _, _, Colors1),
    bird(Name2, _, _, Colors2),
    Name1 \= Name2,
    member(C, Colors1),
    member(C, Colors2).

common_colors(Name1, Name2, Common):-
    bird(Name1, _, _, Colors1),
    bird(Name2, _, _, Colors2),
    findall(C, (member(C, Colors1), member(C, Colors2)), Common).

birds_with_n_colors(N, Birds):-
    findall(Name, (bird(Name, _, _, Colors), length(Colors, N)), Birds).

has_all_colors(Name, RequiredColors):-
    bird(Name, _, _, BirdColors),
    forall(member(C, RequiredColors), member(C, BirdColors)).

has_only_colors(Name, AllowedColors):-
    bird(Name, _, _, BirdColors),
    forall(member(C, BirdColors), member(C, AllowedColors)).

unique_coloring(Name):-
    bird(Name, _, _, Colors),
    \+ (
        bird(OtherName, _, _, OtherColors),
        Name \= OtherName,
        sort(Colors, Sorted),
        sort(OtherColors, Sorted)
    ).

most_diverse_species(Species, Diversity):-
    dif_n_colors(Species, Diversity),
    \+ (
        dif_n_colors(OtherSpecies, OtherDiversity),
        Species \= OtherSpecies,
        OtherDiversity > Diversity
    ).

male_birds_of_species(Species, Males):-
    findall(Name, bird(Name, Species, male, _), Males).

female_birds_of_species(Species, Females):-
    findall(Name, bird(Name, Species, female, _), Females).

gender_ratio(Species, Males-Females):-
    findall(Name, bird(Name, Species, male, _), MaleList),
    findall(Name, bird(Name, Species, female, _), FemaleList),
    length(MaleList, Males),
    length(FemaleList, Females).

dominant_color(Name, Color):-
    bird(Name, _, _, [Color|_]).

same_dominant_color(Name1, Name2):-
    dominant_color(Name1, Color),
    dominant_color(Name2, Color),
    Name1 \= Name2.

color_at_position(Color, Position, Birds):-
    findall(Name, (
        bird(Name, _, _, Colors),
        nth1(Position, Colors, Color)
    ), Birds).

color_frequency(Color, Count):-
    findall(Color, (bird(_, _, _, Colors), member(Color, Colors)), AllOccurrences),
    length(AllOccurrences, Count).

most_common_color(Color):-
    all_colors(Colors),
    member(Color, Colors),
    color_frequency(Color, Count),
    \+ (
        member(OtherColor, Colors),
        color_frequency(OtherColor, OtherCount),
        OtherCount > Count
    ).

least_common_color(Color):-
    all_colors(Colors),
    member(Color, Colors),
    color_frequency(Color, Count),
    \+ (
        member(OtherColor, Colors),
        color_frequency(OtherColor, OtherCount),
        OtherCount < Count
    ).

exact_color_match(RequiredColors, Birds):-
    findall(Name, (
        bird(Name, _, _, BirdColors),
        sort(RequiredColors, SortedRequired),
        sort(BirdColors, SortedRequired)
    ), Birds).

species_with_min_count(MinCount, SpeciesList):-
    findall(Species, (
        all_species(AllSpecies),
        member(Species, AllSpecies),
        count_species(Species, Count),
        Count >= MinCount
    ), SpeciesList).

universal_color_in_species(Species, Color):-
    bird(_, Species, _, _),
    forall(
        bird(Name, Species, _, Colors),
        member(Color, Colors)
    ).

universal_colors(Species, UniversalColors):-
    all_colors(AllColors),
    findall(Color, (
        member(Color, AllColors),
        universal_color_in_species(Species, Color)
    ), UniversalColors).

color_permutation_pairs(Pairs):-
    findall(Name1-Name2, (
        is_color_permutation(Name1, Name2),
        Name1 @< Name2  % Avoid duplicates
    ), Pairs).

distinct_colors_in_species(Species, Count):-
    findall(C, (bird(_, Species, _, Colors), member(C, Colors)), AllColors),
    sort(AllColors, UniqueColors),
    length(UniqueColors, Count).

most_colorful_species(Species, ColorCount):-
    all_species(AllSpecies),
    member(Species, AllSpecies),
    distinct_colors_in_species(Species, ColorCount),
    \+ (
        member(OtherSpecies, AllSpecies),
        distinct_colors_in_species(OtherSpecies, OtherCount),
        OtherCount > ColorCount
    ).

name_contains(Substring, Name):-
    bird(Name, _, _, _),
    atom_chars(Name, NameChars),
    atom_chars(Substring, SubChars),
    append(_, Rest, NameChars),
    append(SubChars, _, Rest).

birds_by_color_count(SortedBirds):-
    findall(Count-Name, (
        bird(Name, _, _, Colors),
        length(Colors, Count)
    ), Pairs),
    sort(0, @>=, Pairs, SortedPairs),
    findall(Name, member(_-Name, SortedPairs), SortedBirds).

balanced_species(Species):-
    gender_ratio(Species, Males-Females),
    Males =:= Females.

unbalanced_species(Species, Males-Females):-
    gender_ratio(Species, Males-Females),
    Males =\= Females.

potential_mates(Name1, Name2):-
    bird(Name1, Species, Gender1, Colors1),
    bird(Name2, Species, Gender2, Colors2),
    Gender1 \= Gender2,
    Name1 @< Name2,
    member(C, Colors1),
    member(C, Colors2).

average_colors_per_species(Species, Average):-
    findall(Count, (
        bird(_, Species, _, Colors),
        length(Colors, Count)
    ), Counts),
    sum_list(Counts, Sum),
    length(Counts, Len),
    Len > 0,
    Average is Sum / Len.

all_different_color_counts(Species):-
    findall(Count, (
        bird(_, Species, _, Colors),
        length(Colors, Count)
    ), Counts),
    sort(Counts, SortedCounts),
    length(Counts, Len1),
    length(SortedCounts, Len2),
    Len1 =:= Len2.
```