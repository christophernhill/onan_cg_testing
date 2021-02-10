using Oceananigans
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.BoundaryConditions

using CUDA
import Oceananigans.Utils: launch!

using KernelAbstractions
using Oceananigans.Architectures: device, CPU, GPU, @hascuda

arch = CPU()
@hascuda arch = GPU()

@kernel function ∇²!(grid, f, ∇²f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
end

Lx, Ly, Lz = 1e3, 1e3, 1
Nx, Ny, Nz =  35,  35, 1
grid = RegularCartesianGrid(size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz) )
x = TracerFields((:c,), arch, grid)
div_U=x.c
# 
struct PCGSolver{A, S}
        architecture :: A
            settings :: S
end

function PCGSolver(;arch=arch, parameters=parameters)
          bcs   = parameters.Template_field.boundary_conditions
          grid  = parameters.Template_field.grid
          maxit = parameters.maxit
          tol   = parameters.tol
          a_res = similar(parameters.Template_field.data);a_res.=0.
          q     = similar(parameters.Template_field.data)
          p     = similar(parameters.Template_field.data)
          z     = similar(parameters.Template_field.data)
          r     = similar(parameters.Template_field.data)
          if parameters.PCmatrix_function == nothing
            # preconditioner not provided, use the Identity matrix
            PCmatrix_function(x) = ( return x )
          else
            PCmatrix_function = parameters.PCmatrix_function
          end
          ii=grid.Hx:grid.Nx+grid.Hx-1
          ji=grid.Hy:grid.Ny+grid.Hy-1
          ki=grid.Hz:grid.Nz+grid.Hz-1
          dotproduct(x,y)  = mapreduce((x,y)->x*y, + , x[ii,ji,ki], y[ii,ji,ki])
          norm(x)          = ( mapreduce((x)->x*x, + , x[ii,ji,ki]   ) )^0.5
          Amatrix_function = parameters.Amatrix_function
          A(x) = ( Amatrix_function(x,a_res,bcs); return  a_res )
          M(x) = ( PCmatrix_function(x) )
          settings = (q=q, 
                      p=p,
                      z=z,
                      r=r,
                    bcs=bcs, 
                   grid=grid,
                      A=A,
                      M=M,
                  maxit=maxit,
                    tol=tol,
                dotprod=dotproduct,
                   norm=norm,
                   arch=arch,
          )

   return PCGSolver(arch, settings)
end

function solve_poisson_equation!(solver::PCGSolver,RHS,x)
#
# Alg - see Fig 2.5 The Preconditioned Conjugate Gradient Method in
#                    "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods"
#                    Barrett et. al, 2nd Edition. 
#
#     given 
#        linear Preconditioner operator M as a function M()
#        linear A matrix operator A as a function A()
#        a dot product function norm()
#        a right-hand side b
#        an initial guess x
#
#        local vectors: z, r, p, q
#
#     β  = 0
#     r .= b-A(x)
#     i  = 0
#
#     loop: 
#      if i > MAXIT 
#       break
#      end
#      z = M( r )
#      ρ    .= dotprod(r,z)
#      p = z+β*p
#      q = A(p)
#      α=ρ/dotprod(p,q)
#      x=x.+αp
#      r=r.-αq
#      if norm2(r) < tol
#       break
#      end
#      i=i+1
#      ρⁱᵐ1 .= ρ
#      β    .= ρⁱᵐ¹/ρ
#
      sset       = solver.settings
      z, r, p, q = sset.z, sset.r, sset.p, sset.q
      A          = sset.A
      M          = sset.M
      maxit      = sset.maxit
      tol        = sset.tol
      dotprod    = sset.dotprod
      norm       = sset.norm

      β    = 0.
      r   .= RHS .- A(x)
      i    = 0
      ρ    = 0
      ρⁱᵐ¹ = 0

      while true
       if i > maxit
        break
       end
       z    .= M(r)
       ρ     = dotprod(z,r)
       if i == 0
        p   .= z
       else
        β    = ρ/ρⁱᵐ¹
        p   .= z .+ β .* p
       end
       q    .= A(p)
       α     = ρ/dotprod(p,q)
       x    .= x .+ α .* p
       r    .= r .- α .* q
       println("Solver ", i," ", norm(r) )
       if norm(r) <= tol
        break
       end
       i     = i+1
       ρⁱᵐ¹  = ρ
      end
#==
#     No preconditioner verison
      i    = 0
      r   .= RHS .- A(x)
      p   .= r
      γ    = dotprod(r,r)
      while true
       if i > maxit
        break
       end
       q   .= A(p)
       α    = γ/dotprod(p,q)
       x   .= x .+ α .* p
       r   .= r .- α .* q
       println("Solver ", i," ", norm(r) )
       if norm(r) <= tol
        break
       end
       γ⁺   = dotprod(r,r)
       β    = γ⁺/γ
       p   .= r .+ β .* p
       γ    = γ⁺
       i    = i+1
      end
==#

      fill_halo_regions!(x, sset.bcs, sset.arch, sset.grid)
      return x, norm(r)
end

function Base.show(io::IO, solver::PCGSolver)
        print(io, "Oceanigans compatible preconditioned conjugate gradient solver.\n")
        print(io, " Problem size = "  , size(solver.settings.q) ) 
        print(io, "\n Boundary conditions = "  , solver.settings.bcs  ) 
        print(io, "\n Grid = "  , solver.settings.grid  ) 
  return nothing
end

function fhr!(x,bcs)
   fill_halo_regions!(x, bcs, arch, grid)
end
function Amatrix_function(x,result,bcs)
 event = launch!(arch, grid, :xyz, ∇²!, grid, x, result, dependencies=Event(device(arch)))
 wait(device(arch), event)
 fhr!(result,bcs)
end


pcg_solver=PCGSolver( ;arch=arch, 
              parameters=(PCmatrix_function=nothing,
                          Amatrix_function= Amatrix_function,
                          Template_field=div_U,
                          maxit=grid.Nx*grid.Ny,
                          tol=1.e-16,
                         )
           )

x=similar(div_U.data);x.=0
div_U.data.=0.
imid=Int(floor(grid.Nx/2))+1
jmid=Int(floor(grid.Ny/2))+1
div_U.data[imid-1,jmid,1]=-1
div_U.data[imid,jmid,1]= 1
fhr!(div_U.data,div_U.boundary_conditions)
solve_poisson_equation!(pcg_solver,div_U.data,x)
