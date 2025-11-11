twinPrimes :: [(Integer,Integer)]
twinPrimes (x,y) = filter (\(x,y)->y==x+2) lst
    where lst = zip primes (tail primes)