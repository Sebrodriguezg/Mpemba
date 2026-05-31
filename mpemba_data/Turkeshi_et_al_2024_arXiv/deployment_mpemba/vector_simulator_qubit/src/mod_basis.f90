module mod_basis 
use, intrinsic :: iso_fortran_env, dp =>real64 
use mod_utils 
implicit none 


! Hilbert space basis class 
type, public :: basis 
  integer :: lsys, nup, size
  integer, allocatable, dimension(:) :: basis_states 
  integer, allocatable, dimension(:) :: state_indices 
contains 
  final :: basis_destructor 
end type basis 

interface basis 
  module procedure basis_constructor_u1
  module procedure basis_constructor_full
end interface basis

contains 

! constructor class for symmetric basis
type(basis) function basis_constructor_u1(lsys,nup) result(this)
  integer :: lsys, nup 
  integer :: size, nst, a, b, c, i
  size = binomial(lsys,nup)
  this%lsys = lsys 
  this%nup = nup 
  this%size = size
  if (allocated(this%state_indices)) deallocate(this%state_indices)
  if (allocated(this%basis_states)) deallocate(this%basis_states)
  allocate(this%state_indices(size),this%basis_states(size))
  a = 2**nup-1 
  nst = 0 
  if (a==0) then 
    nst = nst +1 
    this%state_indices(nst) = 1 
    this%basis_states(nst) = a 
    return 
  end if
  do 
    nst = nst +1 
    this%state_indices(nst) = nst 
    this%basis_states(nst) = a
    c = 0
    do b = 1, lsys 
      if (btest(a,b-1)) then 
        if (btest(a,b)) then 
          c = c+1 
        else 
          do i=0,c-1
            a = ibset(a,i)
          end do 
          do i=c,b-1
            a = ibclr(a,i)
          end do 
          a = ibset(a,b)
          exit 
        end if 
      end if 
    end do 
    if (b==lsys) then 
      exit
    end if 
  end do 
end function basis_constructor_u1

! constructor class for full basis
type(basis) function basis_constructor_full(lsys) result(this)
  integer :: lsys
  integer :: size, i
  size = 2**lsys 
  this%lsys = lsys 
  this%size = size
  if (allocated(this%state_indices)) deallocate(this%state_indices)
  if (allocated(this%basis_states)) deallocate(this%basis_states)
  allocate(this%state_indices(size),this%basis_states(size))
  do i=1,size 
    this%basis_states(i) = i-1 
    this%state_indices(i) = i 
  end do 
end function basis_constructor_full

! destructor class
subroutine basis_destructor(this)
  type(basis) :: this
  if (allocated(this%basis_states)) deallocate(this%basis_states)
  if (allocated(this%state_indices)) deallocate(this%state_indices)
end subroutine basis_destructor 



end module mod_basis
