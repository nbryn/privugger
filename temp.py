def analyze_works(ages):
    if ages[0] < 35:
        subset = ages[:20]
        avg = subset.sum() / subset.size
        return avg
    
    if ages[0] < 40:
        subset = ages[:40]
        avg = subset.sum() / subset.size
        return avg + 2
    
    elif ages[0] < 45:
        subset = ages[:60]
        avg = subset.sum() / subset.size
        return avg
            
    return ages.sum() / ages.size
