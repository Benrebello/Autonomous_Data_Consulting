# tools/advanced_math.py
"""Advanced mathematical operations: linear algebra and optimization.

Provides functions for:
- Linear systems solving
- Eigenvalue/eigenvector computation
- Linear programming optimization
"""

import numpy as np
from typing import List, Optional, Dict, Any


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve linear system Ax = b for x.
    
    Args:
        A: Coefficient matrix (2D list)
        b: Right-hand side vector
    
    Returns:
        Solution vector x as list
    """
    A_array = np.array(A)
    b_array = np.array(b)
    
    try:
        x = np.linalg.solve(A_array, b_array)
        return x.tolist()
    except np.linalg.LinAlgError:
        return []


def compute_eigenvalues_eigenvectors(matrix: List[List[float]]) -> Dict[str, Any]:
    """Compute eigenvalues and eigenvectors of a matrix.
    
    Args:
        matrix: Square matrix (2D list)
    
    Returns:
        Dictionary with eigenvalues and eigenvectors
    """
    matrix_array = np.array(matrix)
    
    try:
        eigvals, eigvecs = np.linalg.eig(matrix_array)
        return {
            'eigenvalues': eigvals.tolist(),
            'eigenvectors': eigvecs.tolist()
        }
    except Exception:
        return {'error': 'Failed to compute eigenvalues/eigenvectors'}


def linear_programming(c: List[float], A_ub: List[List[float]], b_ub: List[float],
                      A_eq: Optional[List[List[float]]] = None, 
                      b_eq: Optional[List[float]] = None) -> Dict[str, Any]:
    """Solve linear programming problem: minimize c^T x subject to constraints.
    
    Args:
        c: Coefficients of objective function
        A_ub: Inequality constraint matrix (A_ub @ x <= b_ub)
        b_ub: Inequality constraint bounds
        A_eq: Optional equality constraint matrix (A_eq @ x == b_eq)
        b_eq: Optional equality constraint bounds
    
    Returns:
        Dictionary with solution status, optimal x, and objective value
    """
    from scipy.optimize import linprog
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
        return {
            'success': bool(res.success),
            'x': res.x.tolist() if res.x is not None else None,
            'fun': float(res.fun) if res.fun is not None else None,
            'message': res.message
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
