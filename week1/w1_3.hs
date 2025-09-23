second :: [a] -> a
second xs = head(tail xs)

last :: [a] -> a
last xs = head(reverse xs)

init :: [a] -> [a]
init xs = reverse(tail(reverse xs))

middle :: [a] -> a
middle xs = head(drop(length xs `div` 2) xs)

checkPalindrome :: Eq a => [a] -> Bool
checkPalindrome str = str == reverse str