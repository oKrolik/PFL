forte :: String -> Bool
forte str = length str >= 8 
    && or [x >= 'A' && x <= 'Z' | x <- str]
    && or [x >= 'a' && x <= 'z' | x <- str]
    && or [x >= '0' && x <= '9' | x <- str]