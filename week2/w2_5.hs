safetailGuards :: [a] -> [a]
safetailGuards xs
    | null xs = []
    | length xs == 1 = xs
    | otherwise = tail xs

safetailIf :: [a] -> [a]
safetailIf xs = if null xs then []
                else if length xs == 1 then xs
                else tail xs

safetailPatterns :: [a] -> [a]
safetailPatterns [] = []
safetailPatterns [x] = [x]
safetailPatterns (x:xs) = xs