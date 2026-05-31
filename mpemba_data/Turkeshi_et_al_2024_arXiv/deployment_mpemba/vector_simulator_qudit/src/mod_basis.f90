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
  procedure :: find_vector
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
  integer :: sc, si, sa 
  size = binomial(lsys,nup)*2**lsys
  this%lsys = lsys 
  this%nup = nup 
  this%size = size
  if (allocated(this%state_indices)) deallocate(this%state_indices)
  if (allocated(this%basis_states)) deallocate(this%basis_states)
  allocate(this%state_indices(size),this%basis_states(size))
  a = 2**nup-1 
  nst = 0 
  if (a==0) then 
    do sc=0,2**lsys-1
      nst = nst +1 
      sa = 0
      do si=0,lsys-1
        if (btest(sc,si)) sa = ibset(sa,2*si+1)
      end do 
      this%state_indices(nst) = nst
      this%basis_states(nst) = sa
    end do  
    return
  end if
  do 
    do sc=0,2**lsys-1
        nst = nst +1 
        sa = 0
        do si=0,lsys-1
          if (btest(sc,si)) sa = ibset(sa,2*si+1)
          if (btest(a,si)) sa = ibset(sa,2*si)
        end do 
        this%state_indices(nst) = nst
        this%basis_states(nst) = sa
    end do
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
  size = 4**lsys 
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

! binary search to find vector in basis
integer function find_vector(this,sa) result(a)
  class(basis), intent(in) :: this 
  integer, intent(in) :: sa
  integer :: amin, amax 
  associate(size=>this%size,state=>this%basis_states)
  amin = 1
  amax = size
  do
    a = amin+(amax-amin)/2
    if (sa< state(a)) then 
      amax = a-1
    else if (sa>state(a)) then 
      amin = a+1
    else 
      return 
    end if
  end do 
  end associate
end function find_vector


end module mod_basis
