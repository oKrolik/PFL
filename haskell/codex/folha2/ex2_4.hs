xor :: Bool -> Bool -> Bool
xor True True = False
xor False False = False
xor True False = True
xor False True = True