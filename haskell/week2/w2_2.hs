classifyBMI :: Float -> Float -> String
classifyBMI weight height
    | bmi < 18.5 = "underweight"
    | bmi < 25 = "normal weight"
    | bmi < 30 = "overweight"
    | otherwise = "obese"
    where bmi = weight / (height * height)