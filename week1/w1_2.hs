import Distribution.Simple.Program.HcPkg (list)
leftHalf :: [a] -> [a]
leftHalf xs = take (length xs `div` 2) xs

rightHalf :: [a] -> [a]
rightHalf xs = drop (length xs `div` 2) xs
