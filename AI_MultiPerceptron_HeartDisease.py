# Final STEP V2.1 (Output Heart Disease = 1, No Heart Disease = 0 **CONVERT**)

import random
import math           # rumus sigmoid
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORT DATASET
dataset = pd.read_csv('G:\A S2\SEM 1\AI_Pak Ali Ridho\M10\AI_M10_MultiPerceptron_Heart\heartV2.csv')
print (dataset)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Input Data
print("==========================================================> Input Data")

Input = dataset.iloc[:,0:13]
print(Input)
print("-"*100)

g, h = 13+1, 270+1;
I = [[0 for x in range(g)] for y in range(h)]

for j in range (270):
    for k in range (13):
        I[j+1][k+1] = Input.iloc[j,k]
        # =================================================================== Convert Input
        if(I[j+1][k+1] >= 100):
            I[j+1][k+1] = I[j+1][k+1]/100
        elif(I[j+1][k+1] >= 10):
            I[j+1][k+1] = I[j+1][k+1]/10
    print("%.2f "*13 %(I[j][1], I[j][2], I[j][3], I[j][4], I[j][5], I[j][6], I[j][7], I[j][8], I[j][9], I[j][10], I[j][11], I[j][12], I[j][13]))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Output Data
print("==========================================================> Output Data")

Output = dataset.iloc[:,-1]
print(Output)
print("-"*100)

g, h = 13+1, 270+1;
Out = {}

for i in range (270):
     # =================================================================== Convert Output
    if (Output[i] == 2):
        Output[i]  = 1
    elif (Output[i] == 1):
        Output[i] = 0
    Out[i+1] = Output[i]
    print("Out[%d] = " %(i+1), Out[i+1])
print("-"*100, i)

print("==========================================================> Declare Variable")

g, h = 15+1, 15+1;
w = [[0 for x in range(g)] for y in range(h)]
dw = [[0 for x in range(g)] for y in range(h)]

wh = {}
summation = {}
H = {}
Cout = {}
CoutTh = {}

dH = {}
dwH = {}

TrueC = 0
FalseC = 0

#===============
Bias = 1

epoch = 5000+1
miu = 0.1
MSE = 0

# Creates weight input layer
print("==========================================================> Creates weight input layer")

for i in range (13+1):
    for j in range (1, 15+1):
        w[i][j] = random.random()
        print("w[%d][%d] = " %(i, j), w[i][j])
print("-"*100)

# Creates weight hidden layer
print("==========================================================> Creates weight hidden layer")

for i in range (15+1):
    wh[i] = random.random()
    print("wh[%d] = " %i, wh[i])
print("-"*100)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning
print("==========================================================> Learning")

for i in range(1, epoch):    
    for j in range (1, 270+1):                # 270 = Jumlah dataset (270 input)
        # =========================================================================================== Forward 
        
        # =================================================================== Input Layer 
        for k in range (1, 15+1):                 # Jumlah hidden layer
            H[k] = Bias*w[0][k]
            for n in range (1, 13+1):         # (Jumlah Input)
                H[k] = H[k] + (I[j][n]*w[n][k])
            # =================================================================== Sigmoid Output
            H[k] = 1 / (1 + math.exp(-H[k]))            
            
        # =================================================================== Hidden Layer 1
        SH = Bias*wh[0]
        for k in range (1, 15+1):
            SH = SH + (H[k]*wh[k])
        # =================================================================== Sigmoid Output
        C = 1 / (1 + math.exp(-SH))       
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning Performance Analysis
        MSE = MSE + (Out[j] - C)**2
        
        # =========================================================================================== Backward
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Differensial
        dC = C*(1 - C)*(Out[j] - C)          # Output = TargetC
        for k in range (1, 15+1):
            dH[k] = H[k]*(1 - H[k])*wh[k]*dC
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Update Weight Hidden Layer
        H[0] = Bias
        for k in range (15+1):          #Jumlah Hidden layer (Bias(H0), H1 dan H2)
            dwH[k] = miu*H[k]*dC
            # Update Weight
            wh[k] = wh[k] + dwH[k]            
            
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Update Weight Input Layer
        for k in range (13+1):     # 1, 13+1 = 13 (Jumlah Input)
            for m in range (1, 15+1):     # 1,3 = 2 (Jumlah Hidden Layer)
                dw[k][m] = miu*I[j][k]*dH[m]
                # Update Weight
                w[k][m] = w[k][m] + dw[k][m]                
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning Performance Analysis
    MSE = MSE / 270    # 270 = jumlah baris data
    print("%.12f => MSE %d " %(MSE, i))
    #print(MSE)
    MSE = 0

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Final Weight
print()
print("==========================================================> Final Weight")

for i in range (13+1):
    for j in range (1, 15+1):
        print("w[%d][%d] = " %(i, j), w[i][j])
print("-"*100)

for i in range (15+1):
    print("wh[%d] = " %i, wh[i])
print("-"*100)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Check Re-Check

print()
print("="*100)
print("==========================================================> Final Result")

for j in range (1, 270+1):                # 270 = Jumlah dataset (270 input)
    
        # =========================================================================================== Forward        
        # =================================================================== Input Layer 
        for k in range (1,15+1):             # 1,3 = 2 - Jumlah hidden layer
            
            H[k] = Bias*w[0][k]
            for n in range (1, 13+1):   # 1, 13+1 = 13 (Jumlah Input)
                H[k] = H[k] + (I[j][n]*w[n][k])
            # =================================================================== Sigmoid Output
            H[k] = ( 1 / (1 + math.exp(-H[k])))        
        # =================================================================== Hidden Layer 1
        SH = Bias*wh[0]
        for k in range (1, 15+1):
            SH = SH + (H[k]*wh[k])
        # =================================================================== Sigmoid Output
        Cout[j] = 1 / (1 + math.exp(-SH))
        # =================================================================== Output Threshold
        if(Cout[j] <= 0.5):
            CoutTh[j] = 0
        else:
            CoutTh[j] = 1
# ===================================================================> Final Result    
print()

for j in range (1, 270+1):                # 270 = Jumlah dataset (270 input)
    print("%.2f "*13 %(I[j][1], I[j][2], I[j][3], I[j][4], I[j][5], I[j][6], I[j][7], I[j][8], I[j][9], I[j][10],
    I[j][11], I[j][12], I[j][13]), " || Output= %.9f" %(Cout[j]), " || CoutTh= ", CoutTh[j], " || Out Real = ", Out[j])
    # =================================================================== Calculate Error (Positive/Negative)
    if (CoutTh[j] == Out[j]):
        TrueC += 1
    else:
        FalseC += 1
        
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Print Error
Error = FalseC/(TrueC+FalseC)

print()
print("==========================================================> Error")
print("True Data = %d || False Data = %d || Error = %f" %(TrueC, FalseC, Error))
