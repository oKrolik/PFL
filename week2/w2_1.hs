classifyIf, classifyGuards :: Int -> String

classifyGuards n
    | n <= 9 = "failed"
    | n <= 12 = "passed"
    | n <= 15 = "good"
    | n <= 18 = "very good"
    | otherwise = "excellent"

classifyIf n =
    if n <= 9 then "failed"
    else if n <= 12 then "passed"
    else if n <= 15 then "good"
    else if n <= 18 then "very good"
    else "excellent"