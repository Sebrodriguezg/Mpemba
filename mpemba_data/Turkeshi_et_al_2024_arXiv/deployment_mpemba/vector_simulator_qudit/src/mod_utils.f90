module mod_utils 
use, intrinsic :: iso_fortran_env, dp=>real64 
implicit none 

! define constants for the code
complex(dp), public, parameter :: ii = (0._dp,1._dp)
real(dp), public, parameter:: zero = 0._dp 
real(dp), public, parameter :: one = 1._dp 
real(dp), public, parameter :: half = 0.5_dp 
real(dp), public, parameter :: two = 2._dp 
real(dp), public, parameter :: pi =3.141592653589793_dp


contains 

! snippet to fix the random seed for reproducibility 
subroutine init_random_seed(arg)
  integer :: i,n, clock
  integer, dimension(:), allocatable :: seed
  integer(dp), optional :: arg
  call random_seed(size=n)
  allocate(seed(n))
  if (present(arg)) then
    clock = arg
  else
    call system_clock(count=clock)
  end if

  seed = clock+37*(/(i-1,i=1,n)/)
  call random_seed(put=seed)
  deallocate(seed)
end subroutine init_random_seed

! eigenvalue routine for the entanglement computation
subroutine eigvalsh(mat,eig)
  integer :: n
  integer :: info, lwork
  real(dp), dimension(:) :: eig
  complex(dp), dimension(:,:) :: mat
  complex(dp), allocatable, dimension(:,:) :: vec
  complex(dp), allocatable, dimension(:) :: work
  real(dp), allocatable, dimension(:) :: rwork

  n = size(eig)
  lwork = max(2*n*(3+n/2),1)
  allocate(work(1:lwork),rwork(1:3*n),vec(1:n,1:n))
  vec = mat 
  call zheev('N','U',n,vec,n,eig,work,lwork,rwork,info) 
  deallocate(work,rwork,vec)
end subroutine eigvalsh

! entanglement entropy routine for the renyi q
subroutine entanglement_entropy(rho,ent,q) 
  complex(dp), dimension(:,:), intent(in) :: rho
  real(dp), intent(in out) :: ent 
  integer, optional, intent(in) :: q 
  real(dp), allocatable, dimension(:) :: eigs
  integer :: dimv, i, qq
  if (present(q)) then 
    qq = q 
  else 
    qq = 1 
  end if 
  dimv = size(rho(1,:))
  allocate(eigs(dimv))
  
  call eigvalsh(rho,eigs)
  eigs = pack(eigs,eigs>1e-16)
  if (qq==1) then 
    ent = -sum(log(eigs)*eigs)/log(two)
  else 
    ent = log(sum(eigs**real(qq,dp)))/(log(two)*(one-qq))
  end if 
  deallocate(eigs)
end subroutine entanglement_entropy 

! qr decomposition rescaled for Mezzardi's algorithm
! to sample unitary matrices 
subroutine qr_dec_rescaled(m,n,a)
  integer, intent(in) :: m, n 
  complex(dp) :: a(m,n), b(n,n)
  complex(dp), allocatable :: work(:), tau(:)
  integer :: i, j, lda, lwork, info
  lda = m
  lwork = 3*n**2
  allocate(tau(max(1,min(m,n))))
  allocate(work(max(1,lwork)))
  call zgeqrf(m,n,a,lda,tau,work,lwork,info)
  b = zero 
  do i=1, m 
    b(i,i) = a(i,i) / abs(a(i,i))
  end do 
  call zungqr(m,n,n,a,lda,tau,work,lwork,info)
  a = matmul(a,b)
  deallocate(work,tau)
end subroutine qr_dec_rescaled 

! Mezzardi's algorithm arXiv: math-ph/0609050
subroutine generate_cue(n,phi)
  integer, intent(in) :: n
  complex(dp), intent(in out) :: phi(n,n)
  real(dp), allocatable, dimension(:,:) :: x, y, z0, z1
  complex(dp), allocatable, dimension(:,:) :: mat
  allocate(x(n,n),y(n,n),z0(n,n),z1(n,n),mat(n,n))
  call random_number(x)
  call random_number(y)
  z0 = sqrt(-two*log(x))*cos(two*pi*y)
  z1 = sqrt(-two*log(x))*sin(two*pi*y)
  mat = (z0+(0._dp,1._dp)*z1)/sqrt(two)
  call qr_dec_rescaled(n,n,mat)
  phi = mat
  deallocate(x,y,z0,z1,mat)
end subroutine generate_cue

! binomial function
integer(dp) function binomial(n,k) result(binom_)
  integer :: n,k, r
  if (k>n) then 
    binom_ = 0
  else if ((k==0) .or. (k==n)) then 
    binom_ = 1
  else 
    binom_ = 1
    do r = 1, k
      binom_ = binom_*int(n-r+1,dp)/int(r,dp)
    end do 
  end if 
end function binomial


end module mod_utils 
