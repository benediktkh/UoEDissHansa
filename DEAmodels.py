# This file contains the two DEA Models used in the Dissertation
import numpy as np
import pandas as pd
import xpress as xp
xp.setOutputEnabled(False)

# 1. DSBM-Model 
## For a Derivation of this Model see Appendix B of the Dissertation


def DSBMModel(input,output,inputfix = None, outputfix = None, carrygood = None, carrybad = None, carryfree = None, carryfix = None,orientation = "Non-Oriented",names = None,rts = "CRS",return_type = 0):
    '''
    This function returns a dictionary/DataFrame to solve the DSBM Model
    Attributes:
    --------------
        input (list/numpy) : array of dimension m/n/T
        output (list/numpy) : array of dimension s/n/T
        inputfix (list/numpy) optional : array of dimension p/n/T
        outputfix (list/numpy) optional : array of dimension r/n/T
        carrygood (list/numpy) optional : array of dimension ngood/n/T
        carrybad (list/numpy) optional : array of dimension nbad/n/T
        carryfree (list/numpy) optional : array of dimension nfree/n/T
        carrybad (list/numpy) optional : array of dimension nbad/n/T

        names (list) optional : names for the DMUs of length n
        rts (string) optional : either CRS (standard) or VRS
        return_type (int) optional : 0 dictionary, 1 pandas DataFrame
    ------------
    Returns:
        results (dict/pandas): results of the DSBM-Model
    '''

    # Ensure inputs and outputs are numpy arrays
    input = np.array(input)
    output = np.array(output)

    # Obtain Parameters
    n = input.shape[1]  # Number of DMUs
    m = input.shape[0]  # Number of Input Resources
    s = output.shape[0] # Number of Output Resources
    T = input.shape[2] # Time Period Length


    # Create logical values to check viability of function Input
    DMUFlag = input.shape[1] == output.shape[1]
    TimeFlag = input.shape[2] == output.shape[2]

    # Check if optional inputs are present and assign values to parameters
    if inputfix is not None:
        inputfix = np.array(inputfix)
        DMUFlag = DMUFlag and n == inputfix.shape[1]
        TimeFlag = TimeFlag and T == inputfix.shape[2] 
        p = inputfix.shape[0] #  Number of fixed Input Resources
    if outputfix is not None:
        outputfix = np.array(outputfix)
        DMUFlag = DMUFlag and n == outputfix.shape[1]
        TimeFlag = TimeFlag and T == outputfix.shape[2] 
        r = outputfix.shape[0] # Number of fixed Output Resources
    if carrygood is not None:
        carrygood = np.array(carrygood)
        DMUFlag = DMUFlag and n == carrygood.shape[1]
        TimeFlag = TimeFlag and T == carrygood.shape[2] 
        ngood = carrygood.shape[0] # Number of good links
    else:
        ngood = 0   # Ensures that the sum later runs over the empty set
    if carrybad is not None:
        carrybad = np.array(carrybad)
        DMUFlag = DMUFlag and n == carrybad.shape[1]
        TimeFlag = TimeFlag and T == carrybad.shape[2] 
        nbad = carrybad.shape[0]   # Number of bad links
    else:
        nbad = 0    # Ensures that the sum later runs over the empty set
    if carryfix is not None:
        carryfix = np.array(carryfix)
        DMUFlag = DMUFlag and n == carryfix.shape[1]
        TimeFlag = TimeFlag and T == carryfix.shape[2] 
        nfix = carryfix.shape[0]    # Number of fixed links
    if carryfree is not None:
        carryfree = np.array(carryfree)
        DMUFlag = DMUFlag and n == carryfree.shape[1]
        TimeFlag = TimeFlag and T == carryfree.shape[2] 
        nfree = carryfree.shape[0]  # Number of free links
    
    
    # Check Errors
    if not DMUFlag:
        raise ValueError(f"Matrices have different DMU dimensions")

    if not TimeFlag:
        raise ValueError(f"Matrices have different Time dimensions")
    
    if not (names is None):
        names = np.array(names)
        
        if names.shape[0] != input.shape[1]:
            raise ValueError(f"Names for DMUs do not match number of DMUs in Inputs and Outputs: Names = {names.shape[0]} != Input/Output = {input.shape[1]}")

    # Create List of all Scores
    score = []
   

    for o in range(n):
        # Solve DSBM-Problem for this DMU
        prob = xp.problem()

        # Variables 
        LbdBig = np.array([[xp.var(lb= 0) for t in range(T)] for j in range(n)])
        SMinBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(m)])
        SPlusBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(s)])

        if carrygood is not None:
            SGoodBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(ngood)])
            prob.addVariable(SGoodBig)

        if carrybad is not None:
            SBadBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(nbad)])
            prob.addVariable(SBadBig)
        
        if carryfree is not None:
            SFreeBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(nfree)])
            prob.addVariable(SFreeBig)

        # Helper Variables
        delta = xp.var(lb = 0)
        tau = xp.var(lb = -np.inf)
        prob.addVariable(LbdBig,SMinBig,SPlusBig,delta,tau)

        # Differences for different Orientations
        if orientation == "Non-Oriented":
            prob.addConstraint(tau == xp.Sum([delta - 1/(m+nbad)*(xp.Sum([SMinBig[i,t]/input[i,o,t] for i in range(m)])+xp.Sum([SBadBig[i,t]/carrybad[i,o,t] for i in range(nbad)])) for t in range(T)]))
            prob.addConstraint(1 == xp.Sum([delta + 1/(s+ngood)*(xp.Sum([SPlusBig[i,t]/output[i,o,t] for i in range(s)])+xp.Sum([SGoodBig[i,t]/carrygood[i,o,t] for i in range(ngood)])) for t in range(T)]))

        if orientation == "Input":
            prob.addConstraint(delta == 1)
            prob.addConstraint(tau == 1/T*xp.Sum([1 - 1/(m+nbad)*(xp.Sum([SMinBig[i,t]/input[i,o,t] for i in range(m)])+xp.Sum([SBadBig[i,t]/carrybad[i,o,t] for i in range(nbad)])) for t in range(T)]))

        if orientation == "Output":
            prob.addConstraint(delta == 1)
            prob.addConstraint(tau == 1/T*xp.Sum([delta + 1/(s+ngood)*(xp.Sum([SPlusBig[i,t]/output[i,o,t] for i in range(s)])+xp.Sum([SGoodBig[i,t]/carrygood[i,o,t] for i in range(ngood)])) for t in range(T)]))


        # Time interdependency
        if carrybad is not None:
            prob.addConstraint([xp.Sum([carrybad[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carrybad[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(nbad) for t in range(T-1)])
        if carrygood is not None:
            prob.addConstraint([xp.Sum([carrygood[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carrygood[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(ngood) for t in range(T-1)])
        if carryfix is not None:
            prob.addConstraint([xp.Sum([carryfix[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carryfix[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(nfix) for t in range(T-1)])
        if carryfree is not None:
            prob.addConstraint([xp.Sum([carryfree[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carryfree[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(nfree) for t in range(T-1)])

        # Add PPF Constraints
        prob.addConstraint([delta*input[i,o,t] == xp.Sum([input[i,j,t]*LbdBig[j,t] for j in range(n)]) + SMinBig[i,t] for i in range(m) for t in range(T)])

        if inputfix is not None:
            prob.addConstraint([delta*inputfix[i,o,t] == xp.Sum([inputfix[i,j,t]*LbdBig[j,t] for j in range(n)]) for i in range(p) for t in range(T)])

        prob.addConstraint([delta*output[i,o,t] == xp.Sum([output[i,j,t]*LbdBig[j,t] for j in range(n)]) - SPlusBig[i,t] for i in range(s) for t in range(T)])

        if outputfix is not None:
            prob.addConstraint([delta*outputfix[i,o,t] == xp.Sum([outputfix[i,j,t]*LbdBig[j,t] for j in range(n)]) for i in range(r) for t in range(T)])

        if carrygood is not None:
            prob.addConstraint([delta*carrygood[i,o,t] == xp.Sum([carrygood[i,j,t]*LbdBig[j,t] for j in range(n)]) - SGoodBig[i,t] for i in range(ngood) for t in range(T)])

        if carrybad is not None:
            prob.addConstraint([delta*carrybad[i,o,t] == xp.Sum([carrybad[i,j,t]*LbdBig[j,t] for j in range(n)]) + SBadBig[i,t] for i in range(nbad) for t in range(T)])

        if carryfree is not None:
            prob.addConstraint([delta*carryfree[i,o,t] == xp.Sum([carryfree[i,j,t]*LbdBig[j,t] for j in range(n)]) + SFreeBig[i,t] for i in range(nfree) for t in range(T)])

        if carryfix is not None:
            prob.addConstraint([delta*carryfix[i,o,t] == xp.Sum([carryfix[i,j,t]*LbdBig[j,t] for j in range(n)]) for i in range(nfix) for t in range(T)])

        if rts == "VRS":
            prob.addConstraint([xp.Sum([LbdBig[j,t] for j in range(n)]) == delta for t in range(T)])


        # Objective function

        if orientation == "Non-Oriented" or orientation == "Input":
            prob.setObjective(tau, sense = xp.minimize)

            prob.solve()

            taustar = prob.getSolution(tau)
            delta = prob.getSolution(delta)

        if orientation == "Output":
            prob.setObjective(tau, sense = xp.maximize)

            prob.solve()

            taustar = 1/prob.getSolution(tau)
            delta = prob.getSolution()

        if delta == 0:
            raise ValueError("delta is equal to zero")
        
            
        

        # Create List for all time periods
        minlist = []
        maxlist = []
        avglist = []
        gaplist = []

        for tcur in range(T):
            prob = xp.problem()

            # Variables
            LbdBig = np.array([[xp.var(lb = 0) for t in range(T)] for j in range(n)])
            SMinBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(m)])
            SPlusBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(s)])

            if carrygood is not None:
                SGoodBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(ngood)])
                prob.addVariable(SGoodBig)

            if carrybad is not None:
                SBadBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(nbad)])
                prob.addVariable(SBadBig)
            
            if carryfree is not None:
                SFreeBig = np.array([[xp.var(lb = 0) for t in range(T)] for i in range(nfree)])
                prob.addVariable(SFreeBig)

            # Helper Variables
            delta = xp.var(lb = 0)
            tau = xp.var(lb = -np.inf)
            taucur = xp.var(lb = -np.inf)
            prob.addVariable(LbdBig,SMinBig,SPlusBig,delta,taucur)

            # Differences for different Orientations
            if orientation == "Non-Oriented":
                prob.addConstraint(taucur == delta - 1/(m+nbad)*(xp.Sum([SMinBig[i,tcur]/input[i,o,tcur] for i in range(m)])+xp.Sum([SBadBig[i,tcur]/carrybad[i,o,tcur] for i in range(nbad)])) for t in range(T))
                prob.addConstraint(1 == delta + 1/(s+ngood)*(xp.Sum([SPlusBig[i,tcur]/output[i,o,tcur] for i in range(s)])+xp.Sum([SGoodBig[i,tcur]/carrygood[i,o,tcur] for i in range(ngood)])))

                prob.addConstraint(taustar*xp.Sum([delta + 1/(s+ngood)*(xp.Sum([SPlusBig[i,t]/output[i,o,t] for i in range(s)])+xp.Sum([SGoodBig[i,t]/carrygood[i,o,t] for i in range(ngood)])) for t in range(T)]) == 
                                    xp.Sum([delta - 1/(m+nbad)*(xp.Sum([SMinBig[i,t]/input[i,o,t] for i in range(m)])+xp.Sum([SBadBig[i,t]/carrybad[i,o,t] for i in range(nbad)])) for t in range(T)]))

            if orientation == "Input":
                prob.addConstraint(delta == 1)
                prob.addConstraint(taucur == 1 - 1/(m+nbad)*(xp.Sum([SMinBig[i,tcur]/input[i,o,tcur] for i in range(m)])+xp.Sum([SBadBig[i,tcur]/carrybad[i,o,tcur] for i in range(nbad)])))

                prob.addConstraint(taustar == 1/T*xp.Sum([1 - 1/(m+nbad)*(xp.Sum([SMinBig[i,t]/input[i,o,t] for i in range(m)])+xp.Sum([SBadBig[i,t]/carrybad[i,o,t] for i in range(nbad)])) for t in range(T)]))

            if orientation == "Output":
                prob.addConstraint(delta == 1)
                prob.addConstraint(taucur == 1 + 1/(s+ngood)*(xp.Sum([SPlusBig[i,tcur]/output[i,o,tcur] for i in range(s)])+xp.Sum([SGoodBig[i,tcur]/carrygood[i,o,tcur] for i in range(ngood)])))
                prob.addConstraint(1/taustar == 1/T*xp.Sum([1 + 1/(s+ngood)*(xp.Sum([SPlusBig[i,t]/output[i,o,t] for i in range(s)])+xp.Sum([SGoodBig[i,t]/carrygood[i,o,t] for i in range(ngood)])) for t in range(T)]))
           
        
            # Time interdependency
            if carrybad is not None:
                prob.addConstraint([xp.Sum([carrybad[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carrybad[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(nbad) for t in range(T-1)])
            if carrygood is not None:
                prob.addConstraint([xp.Sum([carrygood[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carrygood[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(ngood) for t in range(T-1)])
            if carryfix is not None:
                prob.addConstraint([xp.Sum([carryfix[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carryfix[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(nfix) for t in range(T-1)])
            if carryfree is not None:
                prob.addConstraint([xp.Sum([carryfree[i,j,t]*LbdBig[j,t] for j in range(n)]) == xp.Sum([carryfree[i,j,t]*LbdBig[j,t+1] for j in range(n)]) for i in range(nfree) for t in range(T-1)])

            # Add PPF Constraints
            prob.addConstraint([delta*input[i,o,t] == xp.Sum([input[i,j,t]*LbdBig[j,t] for j in range(n)]) + SMinBig[i,t] for i in range(m) for t in range(T)])

            if inputfix is not None:
                prob.addConstraint([delta*inputfix[i,o,t] == xp.Sum([inputfix[i,j,t]*LbdBig[j,t] for j in range(n)]) for i in range(p) for t in range(T)])

            prob.addConstraint([delta*output[i,o,t] == xp.Sum([output[i,j,t]*LbdBig[j,t] for j in range(n)]) - SPlusBig[i,t] for i in range(s) for t in range(T)])

            if outputfix is not None:
                prob.addConstraint([delta*outputfix[i,o,t] == xp.Sum([outputfix[i,j,t]*LbdBig[j,t] for j in range(n)]) for i in range(r) for t in range(T)])

            if carrygood is not None:
                prob.addConstraint([delta*carrygood[i,o,t] == xp.Sum([carrygood[i,j,t]*LbdBig[j,t] for j in range(n)]) - SGoodBig[i,t] for i in range(ngood) for t in range(T)])

            if carrybad is not None:
                prob.addConstraint([delta*carrybad[i,o,t] == xp.Sum([carrybad[i,j,t]*LbdBig[j,t] for j in range(n)]) + SBadBig[i,t] for i in range(nbad) for t in range(T)])

            if carryfree is not None:
                prob.addConstraint([delta*carryfree[i,o,t] == xp.Sum([carryfree[i,j,t]*LbdBig[j,t] for j in range(n)]) + SFreeBig[i,t] for i in range(nfree) for t in range(T)])

            if carryfix is not None:
                prob.addConstraint([delta*carryfix[i,o,t] == xp.Sum([carryfix[i,j,t]*LbdBig[j,t] for j in range(n)]) for i in range(nfix) for t in range(T)])

            if rts == "VRS":
                prob.addConstraint([xp.Sum([LbdBig[j,t] for j in range(n)]) == delta for t in range(T)])

            # Objectives for Minimizing / Maximizing
            if orientation == "Non-Oriented" or orientation == "Input":
                prob.setObjective(taucur, sense = xp.minimize)

                prob.solve()

                taucurmin = prob.getSolution(taucur)
                delta1 = prob.getSolution(delta)

                prob.setObjective(taucur, sense = xp.maximize)

                prob.solve()

                taucurmax = prob.getSolution(taucur)
                delta2 = prob.getSolution(delta)

            if orientation == "Output":
                prob.setObjective(taucur, sense = xp.minimize)

                prob.solve()

                taucurmin = 1/prob.getSolution(taucur)
                delta1 = prob.getSolution(delta)

                prob.setObjective(taucur, sense = xp.maximize)

                prob.solve()

                taucurmax = 1/prob.getSolution(taucur)
                delta2 = prob.getSolution(delta)
                
            if delta1 == 0 or delta2 == 0:
                raise ValueError("delta is equal to zero")

            if taucurmax < taucurmin:
                taucurmin, taucurmax = taucurmax, taucurmin # Rounding Error Catching
                 

            minlist.append(taucurmin)
            maxlist.append(taucurmax)
            avglist.append((taucurmin+taucurmax)/2)
            gaplist.append(taucurmax-taucurmin)

        # Append results to score
        score.append([taustar,minlist,maxlist,avglist,gaplist])

    results = dict()

    if names is None:
        for i in range(n):
            results[f"DMU{i+1}"] = score[i]
    else:
        for i in range(n):
            results[names[i]] = score[i]

    # Return depending on return_type
    if return_type in [0,"dict","Dict","dictionary","Dictionary"]:
        return results
    if return_type in [1,"pandas","DataFrame","dataframe","pd"]:
        return pd.DataFrame.from_dict(results,orient = "index",columns = ["TotalScore","PeriodScoreMin","PeriodScoreMax","PeriodScoreAvg","AbsGap"])


# 2. SBM-Model 
## For a Derivation of this Model see Appendix B of the Dissertation

def SBMModel(input,output,names = None,orientation = "Non-Oriented",rts = "CRS",return_type = 0):
    '''
    This function returns a dictionary/DataFrame to solve the SBM Model
    Attributes:
    --------------
        input (list/numpy) : array of dimension m/n/T
        output (list/numpy) : array of dimension s/n/T
        
        names (list) optional : names for the DMUs of length n
        rts (string) optional : either CRS (standard) or VRS
        return_type (int) optional : 0 dictionary, 1 pandas DataFrame
    ------------
    Returns:
        results (dict/pandas): results of the SBM-Model
    '''
    # Ensures inputs and outputs are numpy arrays
    input = np.array(input)
    output = np.array(output)

    # Checks validty of inputs
    if input.shape[1] != output.shape[1]:
        raise ValueError(f"Input and Output have different number of DMUs: Input = {input.shape[1]} != Output = {output.shape[1]}")
    
    if not (names is None):
        names = np.array(names)
        
        if names.shape[0] != input.shape[1]:
            raise ValueError(f"Names for DMUs do not match number of DMUs in Inputs and Outputs: Names = {names.shape[0]} != Input/Output = {input.shape[1]}")

    # Obtain Parameters
    n = input.shape[1]  # Number of DMUs
    m = input.shape[0]  # Number of Input Resources
    s = output.shape[0] # Number of Output Resources

    # Create list for Scores
    score = []

   
    for o in range(n):
        prob = xp.problem()

        # Variables
        LbdBig = np.array([xp.var(lb = 0) for i in range(n)])
        SMinBig = np.array([xp.var(lb = 0) for i in range(m)])
        SPlusBig = np.array([xp.var(lb = 0) for i in range(s)])

        # Helper Variables
        delta = xp.var(lb = 0)
        tau = xp.var(lb = -np.inf)

        prob.addVariable(LbdBig,SMinBig,SPlusBig,delta,tau)

        # Differences for different Orientations
        if orientation == "Non-Oriented":
            prob.addConstraint(tau == delta - 1/m * xp.Sum([SMinBig[i]/input[i,o] for i in range(m)]))
            prob.addConstraint(1 == delta + 1/s * xp.Sum([SPlusBig[i]/output[i,o] for i in range(s)]))
            prob.setObjective(tau, sense = xp.minimize)
        
        if orientation == "Input":
            prob.addConstraint(tau == delta - 1/m * xp.Sum([SMinBig[i]/input[i,o] for i in range(m)]))
            prob.addConstraint(delta == 1)
            prob.setObjective(tau, sense = xp.minimize)

        if orientation == "Output":
            prob.addConstraint(tau == delta + 1/s * xp.Sum([SPlusBig[i]/output[i,o] for i in range(s)]))
            prob.addConstraint(delta == 1)
            prob.setObjective(tau, sense = xp.maximize)

        # PPS Constraints
        prob.addConstraint([delta*input[i,o] == (input@LbdBig)[i]+SMinBig[i] for i in range(m)])
        prob.addConstraint([delta*output[i,o] == (output@LbdBig)[i]-SPlusBig[i] for i in range(s)])

        prob.addConstraint(delta >= 0)

        if rts == "VRS":
            prob.addConstraint(xp.Sum([LbdBig[i] for i in range(n)]) == delta)

        
        prob.solve()

        # Add solution to score list
        if orientation in ["Non-Oriented","Input"]:
            score.append(prob.getObjVal())
        if orientation == "Output":
            score.append(1/prob.getObjVal())

        
    
    results = dict()

    # Add results to result disctionary
    if names is None:
        for i in range(n):
            results[f"DMU{i+1}"] = [score[i]]
    else:
        for i in range(n):
            results[names[i]] = [score[i]]

    # Return depending on return_type
    if return_type in [0,"dict","Dict","dictionary","Dictionary"]:
        return results
    if return_type in [1,"pandas","DataFrame","dataframe","pd"]:
        return pd.DataFrame.from_dict(results,orient = "index",columns = ["Score"])

