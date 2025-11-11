-- head, tail, length, take, drop and reverse

second :: [a] -> a
second xs = head (reverse (take 2 xs))
-- second (x:xs) = head xs

last :: [a] -> a
last xs = head (reverse xs)

init :: [a] -> [a]
init xs = take (length xs - 1) xs

middle :: [a] -> a
middle xs = head (drop (length xs `div` 2) xs)

checkPalindrome :: String -> Bool
checkPalindrome str = str == reverse str