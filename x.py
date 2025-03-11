def ilp_2v_dp(c1, c2, a11, a12, b1, a21, a22, b2):
    """
    Solves a 2-variable integer linear programming problem using dynamic programming.

    Maximize:      Z = c1*x1 + c2*x2
    Subject to:    a11*x1 + a12*x2 ≤ b1
                   a21*x1 + a22*x2 ≤ b2
                   x1, x2 ≥ 0 and integer

    :param c1, c2: Coefficients of the objective function
    :param a11, a12, b1: Coefficients and RHS of first constraint
    :param a21, a22, b2: Coefficients and RHS of second constraint
    :return: (optimal_x1, optimal_x2, optimal_value)
    """
    dp = {}  # Dictionary to store feasible states and their objective values
    optimal_state = (0, 0)
    optimal_value = float('-inf')  # For minimization

    # Iterate over feasible values of x1
    for x1 in range((b1 // a11) + 1):  # Limit based on constraint 1
        if a21 * x1 > b2:  # Check feasibility w.r.t. constraint 2
            break
        
        # Iterate over feasible values of x2 for a given x1
        for x2 in range((b1 - a11 * x1) // a12 + 1):
            if a21 * x1 + a22 * x2 > b2:  # Check feasibility for second constraint
                break
            
            # Check if the state satisfies both constraints
            if a11 * x1 + a12 * x2 <= b1 and a21 * x1 + a22 * x2 <= b2:
                Z = c1 * x1 + c2 * x2  # Compute objective function value
                dp[(x1, x2)] = Z  # Store the state and its value
                
                # Track the best solution found
                if Z > optimal_value:
                    optimal_value = Z
                    optimal_state = (x1, x2)

    return optimal_state[0], optimal_state[1], optimal_value


# Example Usage
c1, c2 = 3, 1      # Objective function coefficients
#a11, a12, b1 = 94, 22, 8400  # First constraint: 94x1 + 22x2 ≤ 8400
#a21, a22, b2 = 34, 67, 5400   # Second constraint: 34x1 + 67x2 ≤ 5400
a11, a12, b1 = 26, 67, 12748
a21, a22, b2 = 66, 21, 12176

x1_opt, x2_opt, max_value = ilp_2v_dp(c1, c2, a11, a12, b1, a21, a22, b2)
print(f"Optimal solution: x1 = {x1_opt}, x2 = {x2_opt}, Max Z = {max_value}")

cost = c1*x1_opt + c2*x2_opt
constraint1 = a11 * x1_opt + a12 * x2_opt
constraint2 = a21 * x1_opt + a22 * x2_opt
print(f"{constraint1=} | {constraint2=}")

