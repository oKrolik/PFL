{-
 A module for sets implemented using binary search trees
 Pedro Vasconcelos, 2025
-}
module Set (Set,
            empty, fromList,
            insert, member) where

import Data.List (sort)

data Set a
  = Empty
  | Node a (Set a) (Set a)
  -- opaque type:
  -- don't export the constructors
  -- don't derive show

empty :: Set a
empty = Empty

insert :: Ord a => a -> Set a -> Set a
insert x Empty = Node x Empty Empty
insert x (Node y left right)
  | x < y = Node y (insert x left) right
  | x > y = Node y left (insert x right)
  | otherwise = Node y left right

member :: Ord a => a -> Set a -> Bool
member x Empty = False
member x (Node y left right)
  | x == y = True
  | x < y = member x left
  | otherwise = member x right

fromList :: Ord a => [a] -> Set a
fromList xs = build (sort xs)
   where
     build [] = Empty
     build xs = Node x (build xs') (build xs'')
       where
         k = length xs `div` 2
         xs' = take k xs
         x:xs''= drop k xs

treeMap :: (a -> b) -> Set a -> Set b
treeMap _ Empty = Empty
treeMap f (Node v l r) = Node (f v) (treeMap f l) (treeMap f r)
