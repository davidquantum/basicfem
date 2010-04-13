#First let us learn how to integrate using fixed-order Gaussian quadrature.
#We solve the equation -(q(x)u'(x))' + p(x)u(x)= f(x) equipped with Dirichlet boundary conditions at both interval endpoints.

from scipy.integrate import fixed_quad
from scipy.special.orthogonal import p_roots
from scipy.sparse.linalg.dsolve import spsolve
from scipy import sparse
from numpy import arange, zeros, dot, array, exp
from pylab import plot, figure, savefig, grid, legend, clf, pcolor, spy, axis, matshow

# Reference map transforming (-1,1) to (x1,x2)
def refmap(xi, x1, x2): return (x1 + x2)/2. + xi*(x2-x1)/2.

# Integrate bilinear form in element (x1, x2). Here 
# j and i are local indices of shape functions on ref. element
def a(j, i, q, p, x1, x2, jac, qpts_num, roots, weights):
   u = fn[j]
   v = fn[i]
   du = dfn[j]
   dv = dfn[i]
   result = 0.
   for k in range(0, qpts_num):
       x_ref_k = roots[k]
       x_phys_k = refmap(x_ref_k, x1, x2)
       w_phys_k = weights[k]*jac
       u_phys_k = u(x_ref_k)
       v_phys_k = v(x_ref_k)
       du_phys_k = du(x_ref_k)/jac
       dv_phys_k = dv(x_ref_k)/jac
       result += w_phys_k * (q(x_phys_k)*du_phys_k*dv_phys_k 
                             + p(x_phys_k)*u_phys_k*v_phys_k)
   return result

# Integrate linear form in element (x1, x2). Here 
# i is the local index of shape function on ref. element
def l(i, f, x1, x2, jac, qpts_num, roots, weights):
   v = fn[i]
   dv = dfn[i]
   result = 0.
   for k in range(0, qpts_num):
       x_ref_k = roots[k]
       x_phys_k = refmap(x_ref_k, x1, x2)
       w_phys_k = weights[k]*jac
       v_phys_k = v(x_ref_k)
       dv_phys_k = dv(x_ref_k)/jac
       result += w_phys_k * f(x_phys_k) * v_phys_k
   return result

# Create connectivity arrays
# NOTE: This version is for zero Dirichlet conditions only.
def create_connectivity_arrays(n_elem, max_p):
   if max_p < 1: 
       print "ERROR IN CONNECTIVITY()."
       return
   if n_elem < 1: 
       print "ERROR IN CONNECTIVITY()."
       return
   if max_p == 1 and n_elem <= 1: 
       print "ERROR IN CONNECTIVITY()."
       return  
   con = zeros((n_elem, max_p+1))
   # linear part
   for i in range(0, n_elem):
       if i == 0:                # Dirichlet on the left
           con[i][0] = -1
           con[i][1] = 0
       if i == n_elem-1:         # Dirichlet on the right
           con[i][0] = n_elem-2
           con[i][1] = -2
       if i != 0 and i != n_elem - 1:
            con[i][0] = i-1
            con[i][1] = i
   # higher-order part
   count = n_elem - 1
   for i in range(0, n_elem):
       for p in range(2, max_p+1):
           con[i][p] = count
           count += 1
   return con
          
# Calculate solution value
def solution_value(s, m, mesh, connectivity, elem_subdiv, sol, max_p, n_elem):
   val = 0
   x_ref = -1. + s*2./elem_subdiv
   for j in range (0, max_p+1):       # loop over basis functions
       glob_j = connectivity[m][j]    # global index of shape function j
       if glob_j >= 0: 
           val += sol[glob_j]*fn[j](x_ref)
   return val
          
# Solve the equation -(q(x)u'(x))' + p(x)u(x)= f(x) in interval (a, b) with 
# zero Dirichlet conditions on both ends.
def fem_solve_1d_3(mesh, q, p, f, qpts_num, fn, dfn, max_p, exact_sol_defined, exact_sol):
   # import quadrature points and weights 
   roots, weights = p_roots(qpts_num)
   # number of elements
   n_elem = len(mesh) - 1
   # connectivity arrays
   connectivity = create_connectivity_arrays(n_elem, max_p)
   print "Connectivity arrays:"
   print connectivity
   # matrix size
   size = max_p*n_elem - 1
   print "Ndof =", size
   # initiate empty sparse matrix
   I = []
   J = []
   V = []
   # right-hand side vector
   rhs = zeros(size)
   # assembling matrix and right-hand side (element loop)
   print "Assembling."
   for m in range(0, n_elem):     # loop over elements
       x1 = float(mesh[m])        # element left end point
       x2 = float(mesh[m+1])      # element right end point
       h = float(x2 - x1)         # element length
       jac = (x2-x1)/2.           # derivative of reference map
       for i in range (0, max_p+1):       # loop over test functions
           glob_i = connectivity[m][i]    # global index of shape function i
           if glob_i >= 0: 
               for j in range (0, max_p+1):       # loop over basis functions
                   glob_j = connectivity[m][j]    # global index of shape function j
                   if glob_j >= 0: 
                       val_ij = a(j, i, q, p, x1, x2, jac, qpts_num, roots, weights)
                       I.append(glob_i); J.append(glob_j); V.append(val_ij)
               rhs[glob_i] += l(i, f, x1, x2, jac, qpts_num, roots, weights)
	   if glob_i == -1:
               rhs[0] += l(0, f, x1, x2, jac, qpts_num, roots, weights) - a(index_ua, 0, q, p, x1, x2, jac, qpts_num, roots, weights)
   # solve the matrix problem
   print "Solving."
   mat = sparse.coo_matrix((V,(I,J)),shape=(size,size))
   mat = mat.tocsr()
   rhs = array(rhs)
   #sol, res = gmres(mat, rhs, tol=1.e-8)   # also possible: cg, cgs, qmr, gmres, bicg, bicgstab, ...
   sol = spsolve(mat, rhs)
   # plot solution
   x_array = [mesh[0]]
   elem_subdiv = 20
   y_array = [solution_value(0, 0, mesh, connectivity, elem_subdiv, sol, max_p, n_elem)]
   if exact_sol_defined: exact_array = [exact_sol(mesh[0])]
   for m in range(0, n_elem):     # loop over elements
       x1 = float(mesh[m])        # element left end point
       x2 = float(mesh[m+1])      # element right end point
       h = float(x2 - x1)         # element length
       h0 = h/float(elem_subdiv)
       for s in range(1, elem_subdiv+1):
           x_array.append(x1 + s*h0)
           val = solution_value(s, m, mesh, connectivity, elem_subdiv, sol, max_p, n_elem)
           y_array.append(val)
           if exact_sol_defined: exact_array.append(exact_sol(mesh[m] + s*h0))
   print "Plotting."
   clf()
   matshow(mat.todense())
   savefig("a.png")
   clf()
   # Plot solution
   axis('equal')
   if exact_sol_defined: 
       label = "exact"
       plot(x_array, exact_array, "b-", label=label)
   label = "approx"
   plot(x_array, y_array, "g-", label=label)
   legend()
   grid(True)
   savefig("b.png")
   
# Defining shape functions and derivatives
def fn1(xi): return (1. - xi)/2.
def dfn1(xi): return -1./2.
def fn2(xi): return (1. + xi)/2.
def dfn2(xi): return 1./2.
def fn3(xi): return xi**2 - 1.
def dfn3(xi): return 2.*xi
def fn4(xi): return xi*(xi**2 - 1.)
def dfn4(xi): return 3.*xi**2 - 1.
def fn5(xi): return xi**4 - 2.*xi**2 + 1
def dfn5(xi): return 4*xi**3 - 4.*xi
def ua(xi): return 1.
def ub(xi): return 1.
def dua(xi): return 0.
def dub(xi): return 0.
fn = [fn1, fn2, fn3, fn4, fn5, ua, ub]
dfn = [dfn1, dfn2, dfn3, dfn4, dfn5, dua, dub]
  
index_ua = 5;
index_ub = 6; 
# Equation parameters
K = 10.
def q(x):
   return 1.
def p(x):
   return K**2
def f(x):
   return K**2

# Exact solution (if available)
exact_sol_defined = True
def exact_sol(x):
   return 1. - (exp(K*x) + exp(-K*x)) / (exp(K) + exp(-K)) 

# Geometry
x_left = -1.
x_right = 1.

# Finite element mesh
n_elem = 4
mesh = []
for j in range(0, n_elem+1):
   mesh.append(x_left + j*(x_right-x_left)/n_elem)
   
# Polynomial degree of mesh elements
max_p = 4

# Number of quadrature points per element
qpts_num = max_p + 1
   
# Solve the problem
fem_solve_1d_3(mesh, q, p, f, qpts_num, fn, dfn, max_p, exact_sol_defined, exact_sol) 
