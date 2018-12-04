V0.1 frame

# 1. resource arrangement  

## 1.1. initialize searchers and regions  
      
    1.1.1 assign searcher to its region and their investment  
    1.1.2 initialize the sample solutions

            
## 1.2. initialize the investment and set how long regions have not been searched  



# 2. vision search  

## 2.1 construct sampleV (searcher X sample)  

### used GA crossover method

    identity bit = searcher * 0.7 + sample * 0.3
    
other bits :
$X' = rX + (1-r)Y$
$Y' = (1-r)X + rY$

$r = rand(min,max)$

 * min and max can be change

## 2.2 compute the expected value of all regions of searchers  

### 2.2.1 M_j  
    
### 2.2.2 V_ij  
    i-th searcher in j-th region  
        
### 2.2.3 T_j  

### 2.2.4 expected value

## 2.3 select sampleV to sample

# 3. Marketing Research  

## 3.1 update t_a and t_b  

## 3.2 update the best solutions