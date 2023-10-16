# I. Introduction
# In this tutorial, we will learn how to compute quasi-periodic orbits in the CR3BP.

# Before we start, let's import the required packages.
@time using ThreeBodyProblem # CR3BP systems, equations of motion, plotting recipes, differential corrector
@time using DifferentialEquations.OrdinaryDiffEq # For computing solutions to ODEs
@time using LinearAlgebra # For linear algebra computation
@time using Plots # import the Plots package

## I-i. What is a Quasi-Periodic Orbit (QPO)?
# A quasi-periodic orbit (QPO) is a trajectory that exhibits bounded motion, but is not purely periodic. When we think of a purely periodic orbit, we imagine a closed curve in space. If we start at one point along the orbit, we'll return to that exact same point after one period. We saw that the orbit has one degree of freedom, how far along the period it is. 

# A QPO has two degrees of freedom, meaning that the orbit looks like a closed surface rather than a curve. In this case, the surface is a torus and looks something like a donut. If we start at one point along the orbit, we'll never get back to the exact point again. The orbit rotates around the torus with two frequencies, one longitudinal, and one latitudinal. Let's plot a torus so we have a better picture in our mind's eye.
pyplot()
surface(torus(), lims=(-15,15), colorbar=false) # plot a torus

## I-ii. How can we compute a torus?
# Those familiar with differential correction know how to use it to compute a periodic orbit. We start with an initial guess that almost returns to itself after one period T. We refer to the difference between the initial and final states as the error. For a perfectly periodic orbit, the initial state will return to itself exactly after one period, so the error is zero. We compute the Jacobian of the transformation from time 0 to T and use a pseudo-inverse to find the change in the initial conditions that will reduce the error. Iterating this process a few times normally leads to a very small error and our corrected state corresponds to a periodic orbit!
# Since trajectories on a QPO never return to their initial state, we can't use the same method. But we can do something similar. Imagine taking a slice of the torus, which would look like a circle. Any trajectory that starts on that circle will return to the circle, just not at the exact same point. If we had a circle of points that returned to the same circle after one period, we would have a QPO! This is the central idea. Let's set up a system and see if we can apply the concept.

# II. Start with a Halo Orbit

## II-i. Define the system and initial conditions
# We'll work in the Saturn/Enceladus system. We can find precomputed initial conditions for halo orbits at https://ssd.jpl.nasa.gov/tools/periodic_orbits.html

# Define the Saturn/Enceladus CR3BP system
sys = saturn_enceladus();

# Initial condition of the halo orbit (obtained from https://ssd.jpl.nasa.gov/tools/periodic_orbits.html)
rv₀ = [ 1.002850044069033
                        0
        0.004808592996246
       -0.000000000000001
       -0.005747591930694
       -0.000000000000006]

# Period of the halo orbit
T₀ = 2.445783783357601

## II-ii. Differential Correction
# Let's run this initial condition through the differential corrector, just to make sure it's as accurate as possible. The following code creates a differential_corrector() function.
"""
   differential_corrector(sys::System, rv₀; myconst=3, iter=100, plot=false, t0=0., tf=1., dir=1, tol=1e-12)

Given a sufficiently close guess at a periodic trajectory rv₀, returns the corrected initial condition and period
"""
function differential_corrector(sys::System, rv₀; myconst=3, iter=100, plot=false, t0=0., tf=1., dir=1, tol=1e-12)
   Φ₀ = I(6)

   tspan = (t0, tf)

   # event function
   condition(u, t, integrator) = u[2]
   affect!(integrator) = terminate!(integrator)
   cb = OrdinaryDiffEq.ContinuousCallback(condition, affect!)

   for k = 1:iter
      w₀ = vcat(rv₀, reshape(Φ₀,36,1))

      prob = ODEProblem(CR3BPstm!,w₀,tspan,sys)
      sol = solve(prob, TsitPap8(), reltol=tol, callback=cb)

      w = sol[end]
      rv = w[1:6]
      Φ = reshape(w[7:42],6,6)
      global T = 2*sol.t[end]

      ∂ẋ = 0 - rv[4]
      ∂ż = 0 - rv[6]

      if norm([∂ẋ, ∂ż]) < tol
         break
      end
      if k == iter
         @warn("Differential corrector did not converge")
      end

      ẋ,ẏ,ż,ẍ,ÿ,z̈ = CR3BPdynamics(rv, sys, 0)

      if myconst == 1
         ∂Φ = [Φ[4,3] Φ[4,5];
               Φ[6,3] Φ[6,5]]
         dyad = [ẍ;z̈]*[Φ[2,3] Φ[2,5]]

         ∂z, ∂ẏ = (∂Φ - dyad/ẏ)\[∂ẋ; ∂ż]
         rv₀[3] += ∂z
         rv₀[5] += ∂ẏ
      elseif myconst == 3
         ∂Φ = [Φ[4,1] Φ[4,5];
               Φ[6,1] Φ[6,5]]
         dyad = [ẍ;z̈]*[Φ[2,1] Φ[2,5]]

         ∂x, ∂ẏ = (∂Φ - dyad/ẏ)\[∂ẋ; ∂ż]
         rv₀[1] += ∂x
         rv₀[5] += ∂ẏ
      elseif myconst == 5
         ∂Φ = [Φ[4,1] Φ[4,3];
               Φ[6,1] Φ[6,3]]
         dyad = [ẍ;z̈]*[Φ[2,1] Φ[2,5]]

         ∂x, ∂z = (∂Φ - dyad/ẏ)\[∂ẋ; ∂ż]
         rv₀[1] += ∂x
         rv₀[3] += ∂ẏ
      else
         error("myconst should be 1, 3, or 5")
      end

   end

   return rv₀, T
end

# Use the differential corrector to refine the initial conditions (our μ may be slightly different than that used on the website)
rv₀, T₀ = differential_corrector(sys, rv₀, tf=T₀) # differential_corrector() is a function of ThreeBodyProblem.jl

# Jacobi constant of the halo orbit
C₀ = computeC(rv₀,sys)

## II-iii. Integrate and plot
# Now that we have our system and initial condition defined, we'll integrate the halo orbit and plot it

# We'll integrate the halo orbit and its state transition matrix (STM)
Φ₀ = I(6) # Initialization of the STM, Φ₀ = I
w₀ = [rv₀; reshape(Φ₀,36,1)] # Reshape the matrix into a vector and append it to the state vector
tspan = (0.,T₀) # integrate from 0 to T₀
prob_halo = ODEProblem(CR3BPstm!,w₀,tspan,sys) # CR3BPstm! is our in-place dynamics function for state and STM
halo = solve(prob_halo,TsitPap8(),abstol=1e-12,reltol=1e-12) # solve the problem

# Plot the halo orbit
pxy = plot(halo,idxs=(1,2),label="Halo Orbit",legend=false,xaxis="x",yaxis="y"); # plot the halo orbit in the x-y plane
plot!(sys,prim=false,Lpts=false, lims=:auto);

pyz = plot(halo,idxs=(2,3),label="Halo Orbit",legend=false,xaxis="y",yaxis="z"); # plot the halo orbit in the y-z plane
plot!(sys,prim=false,Lpts=false, lims=:auto, center=[0,0,0]);

pxz = plot(halo,idxs=(1,3),label="Halo Orbit", legend=false,xaxis="x",yaxis="z"); # plot the halo orbit in the x-z plane
plot!(sys,prim=false,Lpts=false, lims=:auto);

# pall = plot(halo,idxs=(1,2,3),legend=false,title="Halo Orbit",label="Halo Orbit"); # plot the halo orbit in 3D
# plot!(pall,sys,planar=false,prim=false,Lpts=false,lims=:auto);

plot_halo = plot(pxy,pyz,pxz,layout=(1,3),legend=:outertop) # plot all of the plots in a 2x2 grid
# plot_halo = plot(pxy,pyz,pxz,pall,layout=(1,4),title="Halo Orbit") # plot all of the plots in a 2x2 grid

# III. Approximate the Invariant Circle
# OK, now we have the halo orbit and its state transition matrix (STM). We want to approximate a quasi-halo orbit around the halo orbit. A quasi-halo orbit lives on the surface of a torus, so it is a 2-dimensional object rather than a 1-dimensional curve like the halo orbit. It's tricky working with surfaces. For example, how do you define how closely two surfaces match each other? We'd much rather work with a curve, so let's take a slice of the torus at one moment in time, say t=0. This slice will be a 1D curve called the invariant circle. We can approximate it by using the monodromy matrix, which is the STM at the end of the integration Φ(T,0)

## III-i. Compute eigenvalues and eigenvectors of the monodromy matrix
# Let's pull out the monodromy matrix from our halo object, then compute and sort the eigenvalues/eigenvectors
wf = halo[end] # The final state vector appended with the final STM
M = reshape(wf[7:end],6,6) # The final STM or monodromy matrix M = Φ(T,0)
# We need to compute the eigenvalue decomposition of the monodromy matrix
λ, V = eigen(M) # λ is a vector of eigenvalues and V is a matrix of eigenvectors
sort_idx = sortperm(λ,by=x->(abs(imag(x)),abs(real(x)))) # Sort the eigenvalues by their imaginary magnitude, then by their real magnitude
V = V[:,sort_idx]
λ = λ[sort_idx]

### What do the eigenvalues look like?
# After sorting, the first two eigenvalues are on the real axis, the next two are nearly identity, and the last two are complex conjugate pairs on the unit circle

# Plot the eigenvalues on the complex plan
plot_eig = scatter(real(λ),imag(λ),legend=false,title="Eigenvalues of the Monodromy Matrix",xaxis="Real",yaxis="Imaginary",aspect_ratio=1,marker=:x);
plot!(plot_eig, circle(), seriestype = [:path,], lw = 0.5, linecolor = :black, label = "Unit Circle") # circle() is a function in ThreeBodyProblem.jl that produces the coordinates of the unit circle
annotate!(plot_eig, real(λ[1]), imag(λ[1]), text("Stable",:top, :right));
annotate!(plot_eig, real(λ[2]), imag(λ[2]), text("Unstable",:bottom, :right));
annotate!(plot_eig, real(λ[3]), imag(λ[3]), text("Identity",:top, :left));
annotate!(plot_eig, real(λ[6]), imag(λ[6]), text("Periodic",:top, :left))
# You can see that there are two eigenvalues on the real axis. They are a reciprocal pair and correspond to stable and unstable motion. There is another pair very near to 1. Every purely periodic orbit has two eigenvalues that are 1, which correspond to motion along the periodic orbit itself. Any deviation from 1 is caused by numerical imprecision. The remaining two eigenvalues are complex conjugate pairs on the unit circle. They correspond to periodic motion around the halo orbit. These are the ones we will use to approximate the QPO.

## III-ii. Perturb our initial point
# Because we sorted the eigenvalues by their imaginary magnitude, we should always be able to use the last one to find periodic motion.

eig_idx = 6 # Index of the eigenvalue we want to use
# Note, sometimes, there are two pairs of eigenvalues on the unit circle. In that case, this method will use the one with the largest imaginary magnitude. If there are two pairs of eigenvalues on the unit circle, then the orbit is stable and there exist two periodic manifolds, so we could compute a 3-torus. But let's not worry about that for now.

# We'll compute a ring of N points that are a small step α away from the initial condition and call those points u₀.
N = 19 # Number of points on the invariant circle (THIS SHOULD BE AN ODD NUMBER!!!)
# N is also the number of frequencies that we will break our u function into
n = 6 # Number of dimensions of the system

θ = 2π*(0:N-1)/N # Angles for the invariant circle
α = 1e-3 # parameter to control the size of the invariant circle
u = [α*(cos(θ[i])*real(V[:,eig_idx]) - sin(θ[i])*imag(V[:,eig_idx])) for i in 1:N] # Initial guess for the invariant circle
plot_u = plot(u, xlabel="X [NON]",ylabel="Y [NON]", zlabel= "Z [NON]", legend=true,label="u",title="Approximate Invariant Circle",linecolor=:blue, marker=:x); # Plot the invariant circle
scatter!(plot_u, [u[1][1]],[u[1][2]],[u[1][3]],label="u[1]",shape=:o,markercolor=:blue) # Plot an "x" on the first point of the invariant circle

### What is an invariant circle again?
# An invariant circle is defined as a set of points that are mapped to themselves under the dynamics
# So if this was the true invariant circle, then integrating this ring of points should return the exact same points
# As an added point of subtlety, the points will have rotated by some angle ρ during that integration. This
# ρ is referred to as the rotation number of the invariant circle.

## III-iii. Approximate rotation number
# We'll use the periodic eigenvalue to guess the rotation number of the invariant circle
ρ = real(-im*log(λ[eig_idx])) # Initial guess (in radians) for the rotation number of the invariant circle
# ρ is simply the angle of the eigenvalue in the complex plane. We'll use it later.

## III-iv. Write as a function
# Let's put this invariant circle business into a function for use later.
"""
   invariant_circle(rv, T, N, sys::System; α=1e-5)

Computes an approximation of the invariant circle around a given initial state `rv` 
for a periodic orbit with given period `T` using `N` points. The algorithm uses 
α as the step size. The system is given by `sys`.
"""
function invariant_circle(rv, T, N, sys::System; α=1e-5)
   Φₜ = monodromy(rv, T, sys) # Compute the monodromy matrix
   λ, V = eigen(Φₜ) # λ is a vector of eigenvalues and V is a matrix of eigenvectors

   sort_idx = sortperm(λ,by=x->(abs(imag(x)),abs(real(x)))) # Sort the eigenvalues by their imaginary magnitude, then by their real magnitude
   λ = λ[sort_idx]
   V = V[:,sort_idx]

   eig_idx = 6 # We want the 6th eigenvalue, which is the one with the largest imaginary part, corresponding to periodic motion
   ρ = real(-im*log(λ[eig_idx])) # Initial guess for the rotation number of the invariant circle

   θ = 2π*(0:N-1)/N # Angles for the invariant circle
   u = [α*(cos(θ[i])*real(V[:,eig_idx]) - sin(θ[i])*imag(V[:,eig_idx])) for i in 1:N] # Initial guess for the invariant circle
   
   return u, ρ
end

# IV. Perform Stroboscopic Mapping
# The stroboscopic mapping integrates the invariant circle for one period and rotates it back by ρ radians. The resulting circle should match up with the initial circle (hence the "invariant" circle).

## IV-i. Integrate the invariant circle
# In Julia, we can use the EnsembleProblem to solve the same problem N times with different initial conditions
function prob_func(prob, i, repeat)
    remake(prob, u0=[rv₀+u[i]; reshape(Φ₀,36,1)]) # perturb rv₀ by the ith point of the invariant circle and use that as the initial condition
end # NOTE: Do not get confused! ODEProblems have an field called "u0" (e.g. prob_halo.u0) and the solved problems have a field called "u" (e.g. halo.u). Don't confuse these with the u₀ and u variables that we are defining here.
prob_qpo = EnsembleProblem(prob_halo, prob_func=prob_func) # ODE problem with an ensemble of trajectories. Note: we integrate it for T₀, the same period as the halo orbit, so we just use prob_halo
qpo = solve(prob_qpo, TsitPap8(), trajectories=N, abstol=1e-12, reltol=1e-12) # solve the problem

uT = [qpo[i].u[end][1:6]-rv₀ for i in 1:N] # Invariant circle after integrating (make sure to subtract the base point rv₀)
plot!(plot_u, uT, marker=:x,legend=true,label="uT",linecolor=:red); # Plot the invariant circle after integrating
scatter!(plot_u, [uT[1][1]],[uT[1][2]],[uT[1][3]],label="uT[1]",shape=:o,markercolor=:red); # Plot the first point of the integrated invariant circle
display(plot_u)
# You can see that the invariant circle doesn't quite match the integrated invariant circle. You can also see the rotation number at work, as the first point of the integrated invariant circle is rotated by about ρ radians. Let's plot this in 2D to see the rotation number at work more clearly.

plot_u2D = plot(u, xlabel="X [NON]",ylabel="Y [NON]", planar=true, marker=:x, legend=true,label="u",title="Approximate Invariant Circle",linecolor=:blue); # Plot the invariant circle
scatter!(plot_u2D, [u[1][1]],[u[1][2]], label="u[1]",shape=:o,markercolor=:blue); # Plot a marker on the first point of the invariant circle
plot!(plot_u2D, uT, planar=true, legend=true,label="uT",linecolor=:red, marker=:x); # Plot the invariant circle after integrating
scatter!(plot_u2D, [uT[1][1]],[uT[1][2]], label="uT[1]",shape=:o,markercolor=:red); # Plot the first point of the integrated invariant circle
plot!(plot_u2D, [0,uT[1][1]], [0,uT[1][2]],linecolor=:black,label="");
plot!(plot_u2D, [0,u[1][1]], [0,u[1][2]],linecolor=:black,label="");
annotate!(plot_u2D, 0, 0, text("ρ",:bottom, :right))
# This is called a stroboscopic map. It takes points x(t) and maps them to the points x(t+T). The stroboscopic map of a periodic orbit is a single point, because x(t) = x(t+nT). But for a quasi-periodic orbit, the stroboscopic map is topologically a circle. Therefore, our differential corrector will have to compare circles. If the initial circle and the integrated circle are the same, then the circle is "invariant" and we have found the quasi-periodic orbit

## IV-ii. Compare Circles
# To truly compare the two circles, we need to rotate the integrated invariant circle back by ρ radians. We'll create a rotation operator R to do this. We can't just use an ordinary rotation matrix, as the invariant circle may be elliptical and lives in 6D phase space rather than 3D position space. Instead, we'll use the discrete Fourier transform the invariant circle in the Fourier domain, apply the rotation there, then transform it back to the real domain.

### Discrete Fourier Transform
# D is the discrete Fourier transform matrix. When we left-multiply our invariant circle points with D we get a matrix of Fourier coefficients.
k = Int(-(N-1)/2):Int((N-1)/2); # Vector of integers from -(N-1)/2 to (N-1)/2. N is odd, so (N-1)/2 is an integer.
D = 1/N*exp.(-im*k*θ'); # make sure to use exp.() rather than exp() to perform elementwise exponentiation.

# We are applying the exponential operator to each component rather than performing the matrix exponential. D is a constant matrix, as k and θ won't change. The D matrix is size NxN, so we would like to choose as small of an N as possible. Let's look at what the D matrix does to the invariant circle.
UT = reduce(vcat,uT') # UT is an Nx6 matrix form of uT
pθ = plot(θ, UT, xticks = ([0:π/2:2*π;], ["0","\\pi/2","\\pi","3 \\pi/2","2\\pi"]),xlabel="θ [rad]",ylabel="",legend=true,title="Angular Domain",label=["x" "y" "z" "dx/dt" "dy/dt" "dz/dt"]); # Plot the result
uT_fourier = D*UT # uT_fourier is the invariant circle in the Fourier domain
pf = plot(k*ρ, abs.(uT_fourier),xlabel="frequency [rad/s]",ylabel="magnitude",legend=true,title="Fourier Domain",label=["x" "y" "z" "ẋ" "ẏ" "ż"]); # Plot the result
plot_fourier = plot(pθ,pf,layout=(1,2),size=(1000,400)) # Plot the two plots side by side

### Rotation Operator
# Now that we have our invariant circle converted to the Fourier domain, let's rotate it backwards by ρ radians. Note, we need to rotate it backwards becasuse ρ is the amount it has rotated forward through integration. We need to reverse this rotation.

# Recall that multiplying by eⁱˣ rotates a number by x radians in the complex plane. We can use this fact to create Q(ρ), which rotates every point backward by kᵢρ radians (or forward by -kᵢρ radians).
Q(ρ) = Diagonal(exp.(-im*k*ρ)) # Q is the rotation operator in the fourier domain. It's a diagonal matrix made up of exponential terms

# We now complete the following steps 
# 1. Multiply by D to enter the Fourier domain
# 2. Multiply by Q(ρ) to rotate backwards by ρ radians
# 3. Multiply by D⁻¹ to exit the Fourier domain back into the real domain
# 4. Constrain the result to be purely real (to get rid of any straggling imaginary parts from numerical imprecision)

# We'll put this all into our big rotation operator R(ρ), which is the rotation operator in the real domain.
R(ρ) = real(inv(D)*Q(ρ)*D) #  We use the similarity transform to convert it to the real domain.

UTR = R(ρ)*UT # Rotate the integrated invariant circle back by ρ₀ radians
uTR = [UTR[i,:] for i in 1:N] # Convert back to a vector of vectors
# uTR is the integrated invariant circle rotated back by ρ radians, so we should be able to compare u₀ with uTR. If they are the same, then we have found the quasi-periodic orbit.

# Let's plot to compare
plot!(plot_u, uTR,legend=true,label="uTR",linecolor=:green,marker=:x); # Plot the invariant circle after integrating and rotating
scatter!(plot_u, [uTR[1][1]],[uTR[1][2]],[uTR[1][3]],label="uTR[1]",shape=:o,markercolor=:green) # Plot the first point of the rotated, integrated invariant circle
display(plot_u)
plot!(plot_u2D, uTR, planar=true, legend=true,label="uT",linecolor=:green, marker=:x); # Plot the invariant circle after integrating and rotating
scatter!(plot_u2D, [uTR[1][1]],[uTR[1][2]], label="uTR[1]",shape=:o,markercolor=:green); # Plot the first point of the integrated/rotated invariant circle
display(plot_u2D)
# We can see that the rotation operator has done its job, as the first point of the integrated invariant circle is now rotated back to the same angle as the first point of the initial invariant circle. 

## IV-iii Write as a Function
# Let's put this whole process into a function that we can use later.
"""
   fourier_operator(N)

Returns the fourier operator D, the vector of indices k, and the vector of angles θ
"""
function fourier_operator(N)
   if iseven(N);  error("N must be odd"); end

   θ = 2π*(0:N-1)/N # Angles for the invariant circle
   k = Int(-(N-1)/2):Int((N-1)/2) # Vector of integers from -(N-1)/2 to (N-1)/2. N is odd, so (N-1)/2 is an integer.
   D = 1/N*exp.(-im*k*θ') # D is a constant matrix, as k and θ won't change
   return D, k, θ
end

"""
   rotation_operator(ρ, N)

Returns the rotation operator R(ρ) and its derivative ∂R/∂ρ(ρ) as well as Q(ρ) and ∂Q/∂ρ(ρ)
"""
function rotation_operator(ρ, N)
   D, k = fourier_operator(N) # D is the fft matrix and k is a vector of indices

   Q(ρ) = Diagonal(exp.(-im*k*ρ)) # Q is the rotation operator in the fourier domain. It's a diagonal matrix made up of exponential terms
   # since multiplying by e^(ikρ) rotates a point by kρ radians, we use e^(-ikρ) to rotate backwards by kρ radians.
   R(ρ) = real(inv(D)*Q(ρ)*D) # R is the rotation operator in the real domain. We use the similarity transform to convert it to the real domain.
   # We need to use real() to make sure each component is real. Multiplying UT by R(ρ) rotates the invariant circle by ρ radians
   ∂Q_∂ρ(ρ) = Diagonal(-im*k.*exp.(-im*k*ρ))
   ∂R_∂ρ(ρ) = real(inv(D)*∂Q_∂ρ(ρ)*D)

   return R, Q, ∂Q_∂ρ, ∂R_∂ρ, D
end

"""
   strob_map(rv₀, u₀, sys::System)
   
Given a base point rv₀ on a periodic orbit and and invariant circle u₀, returns the stroboscopic map uTR and the Jacobi constant
"""
function strob_map(rv₀, u, T, ρ, sys::System)
   N = length(u) # N is the number of points used in the invariant circle
   C = sum([computeC(rv₀+u[i],sys) for i in 1:N])/N # Compute Jacobi Constant of the invariant circle

   # First we integrate the orbit
   Φ₀ = I(6) # Initialization of the STM, Φ₀ = I
   w₀ = [rv₀; reshape(Φ₀,36,1)] # Reshape the matrix into a vector and append it to the state vector
   tspan = (0.,T) # integrate from 0 to T₀
   prob_halo = ODEProblem(CR3BPstm!,w₀,tspan,sys) # CR3BPstm! is our in-place dynamics function for state and STM
   function prob_func(prob, i, repeat) # function that defines the initial condition for each trajectory
      remake(prob, u0=[rv₀+u[i]; reshape(Φ₀,36,1)]) # perturb rv₀ by the ith point of the invariant circle and use that as the initial condition
   end # NOTE: Do not get confused! ODEProblems have an field called "u0" (e.g. prob_halo.u0) and the solved problems have a field called "u" (e.g. halo.u). Don't confuse these with the u₀ and u variables that we are defining here.
   prob_qpo = EnsembleProblem(prob_halo, prob_func=prob_func) # ODE problem with an ensemble of trajectories
   qpo = solve(prob_qpo, TsitPap8(), trajectories=N, abstol=1e-12, reltol=1e-12) # solve the problem

   uT = [qpo[i].u[end][1:6]-rv₀ for i in 1:N] # Invariant circle after integrating (make sure to subtract the base point rv₀)
   UT = reduce(vcat,uT') # UT is an Nx6 matrix form of uT

   R,_,_,_ = rotation_operator(ρ, N) # make sure to use the comma so R doesn't become an array of functions

   UTR = R(ρ)*UT # Rotate the integrated invariant circle back by ρ radians
   uTR = [UTR[i,:] for i in 1:N] # Convert back to a vector of vectors

   return uTR, C, qpo, UT
end

## IV-iii.  Compute Error
# We define the error term u_err as the difference between the initial and integrated/rotated invariant circles. If the norm of the error is above a certain threshold, we'll have to do some differential correction
u_err = uTR-u # Compute the error between the initial and integrated/rotated invariant circles
err = norm(u_err) # size of the error
# That error is small but not small enough. We can see that the invariant circles don't line up perfectly yet. We started with a small perturbation α. If we increase α, the error will increase.
ϵ = α/10 # Set the threshold for the error to be 1/10 of α

# V. Differential Correction
# Now we're going to do some differential correction to refine the invariant circle and reduce the error. We should repeat this step until err < ϵ.
### Differential Correction Review

# As a quick review, let's imagine we have some nonlinear function $y=f(x)$. Let's say we want to find the $x^*$ such that $f(x^*) = 0$. Let's also say that we have a good guess $x_0 \approx x^*$, where $y_0 = f(x_0)$. Since $x_0$ is close to $x^*$, then there exists some small $dx$ such that $x_0 + dx = x^*$. This means that $f(x_0 + dx) = f(x^*) = 0$. If we take the taylor expansion of $f(x)$ we get

# \begin{align}
# f(x + dx) = f(x) + \frac{\partial f}{\partial x} dx + O(dx^2)
# \end{align}

# Let's assume dx is small, so we can ignore the second order terms $O(dx^2)$. This leaves

# \begin{align} 
# f(x_0 + dx) = f(x_0) + \frac{\partial f}{\partial x} dx 
# \end{align}

# Recall that $f(x_0 + dx) = 0$, so let's rewrite this as
# \begin{align}
# 0 = y_0 + \frac{\partial f}{\partial x} dx
# \end{align}

# We can compute $dx$ by taking the inverse of $\frac{\partial f}{\partial x}$
# \begin{align}
# dx = -\frac{\partial f}{\partial x}^{-1}y_0
# \end{align}

# We have an equation for the difference between our initial guess and the true value that we want. This is generalizable to vectors
# \begin{align} 
# d\vec{x} = -J^{-1}\vec{y}
# \end{align}
# where $J = \frac{\partial \vec{y}}{\partial \vec{x}}$ is our Jacobian matrix.

# So, if we can create an error vector $\vec{y}$ and a Jacobian matrix $J$, then we can compute the change $d\vec{x}$ to our initial condition that will minimize the error.
# ## V-i. Construct the Error Vector
# We will construct the error vector as a large column vector made up of the vector differences between each point along uTR and u. We'll also add the difference between the average Jacobi constant of uTR and the desired Jacobi constant $C^*$. Whatever we put into the error vector should approach zero after enough iterations, so we are constraining our final QPO to have the same jacobi constant as the underlying halo orbit
# \begin{align}
# \vec{y} = \begin{bmatrix} \vec{u}^{t,R}_1 - \vec{u}_{1} \\ \vdots \\  \vec{u}^{t,R}_N - \vec{u}_{N} \\ C_{u} - C^* \end{bmatrix}.
# \end{align}

# $C_{u}$ is just the average Jacobi constant of the set $\mathcal{u}= \{\vec{u}_1, \dots, \vec{u}_N \}$
# \begin{align} 
# C_{u} = \frac{1}{N}\sum_{i=1}^N C(\vec{u}_i).
# \end{align}

# In this case we'll set $C^*=C_0$, which is the Jacobi constant of the underlying periodic orbit, but we could set it to another value as long as it is sufficiently close to $C_u$. 
Cᵤ = sum([computeC(rv₀+u[i],sys) for i in 1:N])/N # Compute the average Jacobi constant across the invariant circle
dy = [reduce(vcat,u_err); # reduce(vcat,u_err) turns u_err into one big long vector instead of a vector of vectors 
               Cᵤ - C₀];

### How much does the Jacobi constant vary?
# Let's take a look at the Jacobi constant of the invariant circle
plot_C = plot(θ, [computeC(rv₀+u[i],sys) for i in 1:N], xticks = ([0:π/2:2*π;], ["0","\\pi/2","\\pi","3 \\pi/2","2\\pi"]), xlabel="θ [rad]",ylabel="C [NON]", legend=true, title="Jacobi Constant", label="C(u(θ))", color=:blue)
hline!(plot_C, [C₀], label="C₀", linestyle=:dot, color=:magenta) # horizontal line for halo orbit Jacobi constant C₀
hline!(plot_C, [Cᵤ], label="", linestyle=:dot, color=:blue) # horizontal line for average Jacobi constant Cᵤ of u

## V-ii. Construct the Initial Condition Vector
# We don't need the initial condition vector to do our calculation, but it is good to have it written out for clarity when constructing the Jacobian. 
# Our vector should be made up of things that we can change in order to make our stroboscopic mapping match up perfectly. Therefore, it should be made up of u as well as the rotation number ρ and the period T
# \begin{align}
# \vec{x} = \begin{bmatrix} \vec{u}_1 \\ \vdots \\ \vec{u}_{N} \\ T \\ \rho \end{bmatrix},
# \end{align}
# meaning $d\vec{x}$ will be a vector of small changes to these values.
# ## V-iii. Construct the Jacobian Matrix
# The Jacobian matrix $J = \frac{\partial \vec{y}}{\partial \vec{x}}$ can be written out as block components
# \begin{align} 
# J = \begin{bmatrix} \frac{\partial (u^{t,R} - u)}{\partial u} & \frac{\partial (u^{t,R} - u)}{\partial T} & \frac{\partial (u^{t,R} - u)}{\partial \rho} \\ \frac{\partial (C_u - C_0)}{\partial u} & \frac{\partial (C_u - C_0)}{\partial T} & \frac{\partial (C_u - C_0)}{\partial \rho} \end{bmatrix}
# \end{align}
# Luckily, a lot of these terms are zero: $ \frac{\partial u}{\partial T}, \frac{\partial u}{\partial \rho}, \frac{\partial C_u}{\partial T}, \frac{\partial C_u}{\partial \rho}, \frac{\partial C_0}{\partial u}, \frac{\partial C_0}{\partial T}, \frac{\partial C_0}{\partial \rho} = 0$. Also, it is clear that $\frac{\partial u}{\partial u} = I$ so we can simplify our expression to
# \begin{align}
# J = \begin{bmatrix} \frac{\partial u^{t,R}}{\partial u} - I & \frac{\partial u^{t,R}}{\partial T} & \frac{\partial u^{t,R}}{\partial \rho} \\ \frac{\partial C_u}{\partial u} & 0 & 0 \end{bmatrix} = \begin{bmatrix} J_1 & J_2 & J_3 \\ J_4 & 0 & 0\end{bmatrix}.
# \end{align}
# We'll write out each of these terms. The partial with respect to the initial condition coordinates is
# \begin{align} 
# J_1 = \frac{\partial u^{t,R}}{\partial u} = \left(R(-\rho) \otimes I_6 \right)\tilde{\Phi},
# \end{align}
# where $\otimes$ is the Kronecker product and $\tilde{\Phi}$ is a block-diagonal matrix of the state transition matrices
# \begin{align}
# \tilde{\Phi} = \begin{bmatrix} \Phi_1(T, 0) & 0 & \cdots & 0\\
# 0 & \Phi_2(T, 0) & \cdots & 0 \\
# \vdots & \vdots & \ddots & \vdots\\
# 0 & 0 & \cdots & \Phi_N(T, 0)\end{bmatrix},
# \end{align}

# The partial with respect to period is simply the time derivative of the results of the stroboscopic map $u^{t,R}$
# \begin{align}
# \frac{\partial u^{t,R}}{\partial T} = \begin{bmatrix} \dot{\vec{u}}^{t,R}_1 \\ \vdots \\ \dot{\vec{u}}^{t,R}_N \end{bmatrix}.
# \end{align}

# The partial with respect to $\rho$ can be simplified due to the fact that the only part of the stroboscopic map that depends on $\rho$ is the Fourier rotation operator $Q(\rho)$, which is a diagonal matrix of exponential terms. This gives us
# \begin{align}
# \frac{\partial u^{t,R}}{\partial \rho} = D^{-1} \frac{\partial Q}{\partial \rho} D U^t,
# \end{align}
# where $U^t = \begin{bmatrix} \vec{u}_1^t& \dots & \vec{u}_N^t \end{bmatrix}^\intercal$ and $\frac{\partial Q}{\partial \rho} = \text{diag}(-i k_1 e^{-ik_1\rho},\dots,-i k_N e^{-ik_N\rho})$.

# Finally, the partial of the Jacobi constant with respect to the initial condition coordinates is a long row vector
# \begin{align} 
# \frac{\partial C_u}{\partial u} = \frac{1}{N}\begin{bmatrix} 2\Omega_{x,1} & 2\Omega_{y,1} & 2\Omega_{z,1} & -2\dot{x}_1 & -2\dot{y}_1 & -2\dot{z}_1 &  \dots & 2\Omega_{x,N} & 2\Omega_{y,N} & 2\Omega_{z,N} & -2\dot{x}_N & -2\dot{y}_N & -2\dot{z}_N \end{bmatrix},
# \end{align} 
# where $\Omega_x = \frac{\partial \Omega}{\partial x}$, etc.

### $J₁ = \frac{\partial u^{t,R}}{\partial u} - I$
Φ_tilde = zeros(6N,6N);
for i = 1:N
    idx = (i-1)*n + 1:i*n;
    Φ_tilde[idx,idx] = reshape(qpo[i].u[end][7:end],6,6); # Φ_tilde is made up of the state transition matrices of each point of the invariant circle (unrotated points, as the rotation operator doesn't depend on u and gets multiplied later)
end
∂uTR_∂u = kron(R(ρ),I(6))*Φ_tilde; # Compute the derivative of the integrated/rotated invariant circle with respect to the initial invariant circle (u₀) 
J₁ = ∂uTR_∂u - I(6N); # We subtract the identity because we want ∂(uTR-u)/∂u

### $J₂ = \frac{\partial u^{t,R}}{\partial T}$
J₂ = zeros(6N,1) # column vector of size 6N
for i = 1:N
    idx = (i-1)*n + 1:i*n
    ẋ, ẏ, ż, ẍ, ÿ, z̈ = CR3BPdynamics(rv₀ + uTR[i],sys,0) # don't forget to add rv₀ to uTR[i]
    J₂[idx] = [ẋ, ẏ, ż, ẍ, ÿ, z̈] # The derivative with respect to time comes right from the equations of motion
end

### $J₃ = \frac{\partial u^{t,R}}{\partial \rho}$
∂Q_∂ρ(ρ) = Diagonal(-im*k.*exp.(-im*k*ρ)); # Compute the derivative of the rotation operator in the Fourier domain
J₃ = real(inv(D)*∂Q_∂ρ(ρ)*D)*UT; # Compute the derivative of the rotation operator in the real domain
J₃ = reshape(J₃',6N,1); # Convert to a column vector

### $J₄ = \frac{\partial C}{\partial u}$
J₄ = zeros(1,6N)
for i = 1:N
    idx = (i-1)*n + 1:i*n
    ẋ, ẏ, ż, ẍ, ÿ, z̈ = CR3BPdynamics(rv₀ + u[i],sys,0) # Note, I'm using u instead of uTR here.
    Ωx = ẍ - 2ẏ
    Ωy = ÿ + 2ẋ
    Ωz = z̈
    J₄[idx] = [2Ωx, 2Ωy, 2Ωz, -2ẋ, -2ẏ, -2ż]./N # Don't forget to divide by N because Cᵤ is an average of every C(uᵢ)
end    

### The full Jacobian
J = [J₁  J₂  J₃; # block matrix notation is easy in Julia
     J₄   0   0];

### Add a row to constrain one of the components
# There's one more issue with our differential corrector. It just so happens that the periodic orbit itself is a solution to our scheme. Just imagine the invariant circle as shrinking down to a single point. To keep our differential corrector from converging to the periodic orbit, we must constrain at least one of the components of the original invariant circle. We do this by appending a single row to the Jacobian $[1,0,0,\dots,0]$ and appending a single 0 to the constraint vector. This essentially constrains the x component of the first point of the invariant circle to remain constant 
# \begin{align}
# dx_1 = 0
# \end{align}
row = zeros(1,N*n+2) # add a row to the bottom of J
row[1] = 1 # first element is 1
J = [J;row]; # add the row to the bottom of J
dy = [dy; 0]; # add a zero to the bottom of dy

## V-iv. Compute Optimal Change in Initial Conditions
dx = -J\dy; # the backward slash is the left division operator. It is similar to J⁻¹dy but works even for non-invertible matrices.

# Take out each component
du = [dx[(i-1)*n + 1:i*n] for i = 1:N];
dT = dx[6N+1];
dρ = dx[6N+2];

### Before we apply du, dρ, and dT, let's make sure we save our initial conditions for later
u₀ = u # u₀ is defined as the initial guess for the invariant circle
T = T₀ # T₀ is defined as the period of the halo orbit. We expect that T will vary slightly from T₀
ρ₀ = ρ # ρ₀ is defined as the initial guess for the rotation angle

# Update guesses for u, ρ, and T
u = u + du;
T = T + dT;
ρ = ρ + dρ;

# What does the Jacobi constant profile of the new guess look like?
Cᵤ = sum([computeC(rv₀+u[i],sys) for i in 1:N])/N

# Add new guess for u to the Jacobi constant plot
plot!(plot_C, θ, [computeC(rv₀+u[i],sys) for i in 1:N], xticks = ([0:π/2:2*π;], ["0","\\pi/2","\\pi","3 \\pi/2","2\\pi"]), xlabel="θ [rad]",ylabel="C [NON]", legend=true, title="Jacobi Constant", label="C(u⁽¹⁾(θ))", color=:black)
hline!(plot_C, [Cᵤ], label="", linestyle=:dot, color=:black) # horizontal line for average Jacobi constant Cᵤ of u!
# We can see that the Jacobi Constant of the new guess matches up very nicely with the desired C₀

### What does the new guess look like?
plot!(plot_u, u,legend=true,label="u⁽¹⁾",linecolor=:black); # Plot the invariant circle after integrating
scatter!(plot_u, [u[1][1]],[u[1][2]],[u[1][3]],label="u⁽¹⁾[1]",shape=:o,markercolor=:black); # Plot the first point of the integrated invariant circle
display(plot_u)

plot!(plot_u2D, u, planar=true, legend=true,label="u⁽¹⁾",linecolor=:black, marker=:x); # Plot the invariant circle after integrating and rotating
scatter!(plot_u2D, [u[1][1]],[u[1][2]], label="u⁽¹⁾[1]",shape=:o,markercolor=:black) # Plot the first point of the integrated/rotated invariant circle
display(plot_u2D)
# Our optimizer expanded our initial guess a bit.

## V-v Iterate until Convergence
# Let's see if how the new guess does in the stroposcopic map. If it returns to the initial state closely enough (i.e. the error is small enough), then we have our QPO!
uTR, Cᵤ, qpo_2 = strob_map(rv₀, u, T, ρ, sys);
plot_unew = plot(u,legend=true,label="u⁽¹⁾",color=:black,planar=false,marker=:x); # Plot the invariant circle after integrating
scatter!(plot_unew, [u[1][1]],[u[1][2]],[u[1][3]],label="u⁽¹⁾[1]",shape=:o,color=:black); # Plot the first point of the integrated invariant circle
plot!(plot_unew, uTR,legend=true,label="uTR",color=:green,planar=false,marker=:x); # Plot the invariant circle after integrating
scatter!(plot_unew, [uTR[1][1]],[uTR[1][2]],[uTR[1][3]],label="uTR[1]",shape=:o,color=:green); # Plot the first point of the integrated invariant circle
display(plot_unew)
plot_u2Dnew = plot(u,legend=true,label="u⁽¹⁾",color=:black, planar=true,marker=:x); # Plot the invariant circle after integrating
scatter!(plot_u2Dnew, [u[1][1]],[u[1][2]],label="u⁽¹⁾[1]",shape=:o,color=:black); # Plot the first point of the integrated invariant circle
plot!(plot_u2Dnew, uTR,legend=true,label="uTR",color=:green, planar=true,marker=:x); # Plot the invariant circle after integrating
scatter!(plot_u2Dnew, [uTR[1][1]],[uTR[1][2]],label="uTR[1]",shape=:o,color=:green); # Plot the first point of the integrated invariant circle
display(plot_u2Dnew)

# We can see that the stroboscopic map returns to the exact same circle! We can check our error
u_err = uTR-u # Compute the error between the initial and integrated/rotated invariant circles
err = norm(u_err) # size of the error
println("The error is $(err), so our stopping criterion is $(err < ϵ)")

### Plot the QPO
plot_qpo = plot(sphere(sys.sec.R/sys.RUNIT,[1-sys.μ,0,0]),color=:blue,legend=false)
plot!(plot_qpo, qpo_2, idxs=(1,2,3))
xlabel!("x [NON]")
ylabel!("y [NON]")
zlabel!("z [NON]")
# This QPO is very thin. We can find larger ones through the process of continuation.

## V-vi. Put into a function
# Let's put the whole differential correction code into a function
"""
   monodromy(rv₀, T, sys::System)

computes the monodromy matrix for a given periodic orbit with initial conditions
rv₀ and period T. Returns Φ(T,0).
"""
function monodromy(rv₀, T, sys::System)
   Φ₀ = I(6) # Initialization of the STM, Φ₀ = I
   w₀ = [rv₀; reshape(Φ₀,36,1)] # Reshape the matrix into a vector and append it to the state vector
   tspan = (0.,T) # integrate from 0 to T
   prob = ODEProblem(CR3BPstm!,w₀,tspan,sys) # CR3BPstm! is our in-place dynamics function for state and STM
   sol = solve(prob,TsitPap8(),abstol=1e-12,reltol=1e-12) # solve the problem
   Φₜ = reshape(sol[end][7:end],6,6) # The final STM or monodromy matrix M = Φ(T,0)
   return Φₜ
end
"""
   add_state_constraints!(J, dγ, u; constraints=[])

add the state constraints to the Jacobian and error vector
"""
function add_state_constraints(J, dγ, u; constraints=[])
   N = length(u)
   n = length(u[1])
   for j in constraints
      row = zeros(1,n*N+2) # +2 for the time period and the phase constraint
      row[j] = 1 # Put a 1 in the row of zeros
      J = [J;row] # Add the row to the Jacobian
      append!(dγ,0) # Add a zero to the error vector
   end
   return J, dγ
end


"""
   add_phase_constraints!(J, dγ, u, rv₀, u₀, T₀, ρ₀, N, sys)

Appends the phase constraints for the invariant circle
"""
function add_phase_constraints(J, dγ, u, rv₀, u₀, T₀, ρ₀, N, sys)
   n = length(rv₀)
   # Phase constraints
   D, k, θ = fourier_operator(N) # Compute the fast fourier transform matrix
   C0_tilde = D*reduce(vcat,u₀') # convert to an Nx6 matrix
   ∂u₀_∂θ₁ = im.*exp.(im.*θ*k')*Diagonal(k)*C0_tilde # Derivative of the initial invariant circle with respect to θ₁
   if maximum(imag(∂u₀_∂θ₁)) > eps() # If the derivative is complex, then we have a problem
      error("The derivative of the initial invariant circle with respect to θ₁ is complex")
   end
   ∂u₀_∂θ₁ = real(∂u₀_∂θ₁) # Enforce that the derivative is real
   
   U̇₀ = zeros(N,n)
   for i = 1:N
      U̇₀[i,:] = CR3BPdynamics(rv₀ + u₀[i],sys,0)
   end
   ∂u₀_∂θ₀ = T₀/2π .* (U̇₀ - ρ₀/T₀.*∂u₀_∂θ₁) # Derivative of the initial invariant circle with respect to θ₀
   
   ∂u₀_∂θ₀ = reshape(∂u₀_∂θ₀',1,n*N) # Convert to a row vector
   ∂u₀_∂θ₁ = reshape(∂u₀_∂θ₁',1,n*N) # Convert to a row vector

   J = [J;        # Add the phase constraints to the Jacobian
   ∂u₀_∂θ₀ 0 0;
   ∂u₀_∂θ₁ 0 0]

   append!(dγ,∂u₀_∂θ₀*reduce(vcat,u)) # Add the first phase constraint to the error vector
   append!(dγ,∂u₀_∂θ₁*reduce(vcat,u)) # Add the second phase constraint to the error vector

   return J, dγ
end



"""
   differential_corrector_QPO(sys::System, rv₀, u₀, T₀, ρ₀; max_iter=10, plot_on=false, ϵ=1e-6, constraint::Symbol=:C)

Differential corrector for quasi-periodic orbits QPO problem. 
Takes in a System sys, periodic orbit state rv₀, invariant circle.
Returns the corrected state vector and the time period T.
"""
function differential_corrector_QPO(sys::System, rv₀, u₀, T₀, ρ₀; max_iter=10, plot_on=false, ϵ=1e-6, constraint::Symbol=:C)
   u = u₀ # u₀ is the invariant circle for which the phase constraint will be enforced
   T = T₀ # Initial guess at the period of the QPO
   ρ = ρ₀ # Initial guess at the rotation number of the QPO
   
   N = length(u) # Number of points along invariant circle 
   if iseven(N);  @error "N must be an odd number";   end # Should be an odd number

   n = length(rv₀) # Dimension of state vector (normally n = 6)

   C₀ = computeC(rv₀,sys) # Jacobi constant of central orbit
   C = sum([computeC(rv₀+u[i],sys) for i in 1:N])/N # Compute the average Jacobi constant across the invariant circle

   qpo = []
   uTRs = []
   us = [u]
   Cs = [C]
   # while err > ϵ
   R, _, _, ∂R_∂ρ = rotation_operator(ρ,N) # ∂R_∂ρ is the fourth output from the rotation_operator function
   
   J = zeros(n*N+4,n*N+3) # Initialize the Jacobian to be a matrix of zeros

   for _ in 1:max_iter
      # First we perform the stroboscopic mapping
      uTR, C, qpo, UT = strob_map(rv₀, u, T, ρ, sys) # uTR is the invariant circle after the stroboscopic map
      push!(uTRs,uTR) # Save the invariant circle after the stroboscopic map

      u_err = uTR - u # Compute the error between the initial and integrated/rotated invariant circles
      err = norm(u_err)
      if err < ϵ; # If the error is small enough, we're done
         @info "CONVERGED Differential corrector error = $err"
         break;
      else
         @info "Differential corrector error = $err"
      end

      # The error vector serves as our constraint. We want to drive dγ to zero.
      dγ = [reduce(vcat,u_err); # reduce(vcat,u_err) turns u_err into one big long vector instead of a vector of vectors 
                        C - C₀]
                        
      # The next step is to compute the Jacobian of the stroboscopic map with respect to the initial invariant circle u, rotation number ρ, and period T
      # Let's start with J₁ = ∂(uTR-u₀)/∂u
      Φ_tilde = zeros(n*N,n*N)
      for i = 1:N
         idx = (i-1)*n + 1:i*n
         Φ_tilde[idx,idx] = reshape(qpo[i].u[end][7:end],6,6) # Φ_tilde is made up of the state transition matrices of each point of the invariant circle (unrotated points, as the rotation operator doesn't depend on u and gets multiplied later)
      end
      ∂uTR_∂u = kron(R(ρ),I(6))*Φ_tilde # Compute the derivative of the integrated/rotated invariant circle with respect to the initial invariant circle (u₀) 
      J₁ = ∂uTR_∂u - I(n*N) # We subtract the identity because we really want ∂(uTR-u)/∂u

      # Next  J₂ = ∂uTR/∂T
      J₂ = zeros(n*N,1) # column vector of size n*N
      for i = 1:N
         idx = (i-1)*n + 1:i*n
         ẋ, ẏ, ż, ẍ, ÿ, z̈ = CR3BPdynamics(rv₀ + uTR[i],sys,0) # don't forget to add rv₀ to uTR[i]
         J₂[idx] = [ẋ, ẏ, ż, ẍ, ÿ, z̈] # The derivative with respect to time comes right from the equations of motion
      end
      
      # Next  J₃ = ∂uTR/∂ρ
      J₃ = ∂R_∂ρ(ρ)*UT # Compute the derivative of the rotation operator in the real domain
      J₃ = reshape(J₃',n*N,1) # Convert to a column vector

      # Finally J₄ = ∂C/∂u
      J₄ = zeros(1,n*N) # row vector of size n*N
      for i = 1:N
         idx = (i-1)*n + 1:i*n
         ẋ, ẏ, ż, ẍ, ÿ, z̈ = CR3BPdynamics(rv₀ + u[i],sys,0) # use u[i] here instead of uTR[i]
         Ωx = ẍ - 2ẏ
         Ωy = ÿ + 2ẋ
         Ωz = z̈
         J₄[idx] = [2Ωx, 2Ωy, 2Ωz, -2ẋ, -2ẏ, -2ż]/N # We divide by N because we want the derivative of C_avg with respect to u
      end

      J = [J₁  J₂  J₃;
           J₄   0   0];

      # J, dγ = add_phase_constraints(J, dγ, u, rv₀, u₀, T₀, ρ₀, N, sys)
      J, dγ = add_state_constraints(J, dγ, u, constraints = [1])

      dξ = -J\dγ
      du = [dξ[(i-1)*n + 1:i*n] for i = 1:N]
      dT = dξ[n*N+1]
      dρ = dξ[n*N+2]

      u += du
      T += dT
      ρ += dρ

      C = sum([computeC(rv₀+u[i],sys) for i in 1:N])/N # Compute the Jacobi constant for each state along the invariant circle

      push!(us,u)
      push!(Cs,C)

   end

   _,_,V = svd(J)
   return u, T, ρ, C, us, uTRs, Cs, qpo, V[:,end]
end

# If we run the differential_corrector_QPO code, we should get the same result.
u, T, ρ, C, us, uTRs, Cs, qpo = differential_corrector_QPO(sys,rv₀,u₀,ρ₀,T₀,max_iter = 10);
plot_qpo = plot(sphere(sys.sec.R/sys.RUNIT,[1-sys.μ,0,0]),color=:blue,legend=false)
plot!(plot_qpo, qpo, idxs=(1,2,3), xlabel="x [NON]", ylabel="y [NON]", zlabel="z [NON]")

### Let's try making a much larger QPO
# Approximate a new invariant circle with a larger α value
u,ρ = invariant_circle(rv₀, T₀, N, sys; α=1e-3) # change α to 1e-3
u, T, ρ, C, us, uTRs, Cs, qpo, Δs = differential_corrector_QPO(sys,rv₀,u,ρ,T₀,max_iter = 10, ϵ=1e-12);
plot_qpo = plot(sphere(sys.sec.R/sys.RUNIT,[1-sys.μ,0,0]),color=:blue,legend=false)
plot!(plot_qpo, qpo, idxs=(1,2,3), xlabel="x [NON]", ylabel="y [NON]", zlabel="z [NON]",title="QPO with α = 1e-3")
plot_Tρ = scatter([T],[ρ],xlabel="T",ylabel="ρ")

# using MAT
using JLD2
save("qpo4.jld2",Dict(
   "qpo" => qpo,
   "u" => u,
   "T" => T,
   "C" => C,
   "rho" => ρ,
   "us" => us,
   "uTRs" => uTRs,
   "Cs" => Cs,
   "delta_s" => Δs
))

β = 1e-2

ξ₁ = vcat(reduce(vcat,u),T,ρ) + β*Δs
u = [ξ₁[(i-1)*n + 1:i*n] for i = 1:N]
T = ξ₁[n*N+1]
ρ = ξ₁[n*N+2]
u, T, ρ, C, us, uTRs, Cs, qpo, Δξ = differential_corrector_QPO(sys,rv₀,u,ρ,T₀,max_iter = 10, ϵ=1e-12);
plot!(plot_qpo, qpo, idxs=(1,2,3), xlabel="x [NON]", ylabel="y [NON]", zlabel="z [NON]",title="QPO with α = 1e-3")

scatter!(plot_Tρ,[T],[ρ])


qpo = load("qpo4.jld2","qpo")
plot_qpo = plot(sphere(sys.sec.R/sys.RUNIT,[1-sys.μ,0,0]),color=:blue,legend=false)
plot!(plot_qpo, qpo, idxs=(1,2,3), xlabel="x [NON]", ylabel="y [NON]", zlabel="z [NON]",title="QPO with α = 1e-3")
