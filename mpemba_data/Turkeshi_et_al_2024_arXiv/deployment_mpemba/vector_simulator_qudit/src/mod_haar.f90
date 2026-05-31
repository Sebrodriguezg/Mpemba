module mod_haar
use, intrinsic :: iso_fortran_env, dp=>real64 
use mod_utils
use mod_basis
implicit none 


! circuit class 
type, extends(basis) :: circuit 
  real(dp) :: prob 
  integer :: na, dimv, dimc 
  complex(dp), allocatable, dimension(:) :: temp, psi  
  complex(dp), allocatable, dimension(:,:) :: rho_a, rho_aq 
  complex(dp), dimension(16,16) :: gate
  integer, allocatable, dimension(:) :: shiftr, shiftl, shift2r
  type(basis), allocatable, dimension(:) :: idx_rho_aq, idx_gate
contains 
  procedure :: set_initial_theta_state
  procedure :: set_initial_theta_neel
  procedure :: get_density_matrix
  procedure :: get_density_matrix_symproj

  procedure :: apply_u1_unitary_layer 
  procedure :: apply_u1_gate_and_shift 

  procedure :: generate_gate
  procedure :: get_asym_distances
  procedure :: get_asym 
  final :: circuit_destructor 
end type circuit 

interface circuit 
  module procedure circuit_constructor
end interface circuit 

contains 

! circuit contructor 
function circuit_constructor(lsys,prob,na) result(self)
  type(circuit) :: self 
  type(basis) :: parent  
  integer :: lsys, na 
  real(dp) :: prob 
  integer :: size, s, nst, i
  parent = basis(lsys)
  self%lsys = parent%lsys 
  self%size = parent%size 
  self%prob = prob 
  self%dimv = 4**na 
  self%dimc = 4**(lsys-na) 
  self%na = na 
  if (allocated(self%idx_rho_aq)) deallocate(self%idx_rho_aq)
  if (allocated(self%idx_gate)) deallocate(self%idx_gate)
  if (allocated(self%shiftr)) deallocate(self%shiftr)
  if (allocated(self%shiftl)) deallocate(self%shiftl)
  if (allocated(self%shift2r)) deallocate(self%shift2r)
  if (allocated(self%rho_a)) deallocate(self%rho_a)
  if (allocated(self%rho_aq)) deallocate(self%rho_aq)
  if (allocated(self%psi)) deallocate(self%psi)
  if (allocated(self%temp)) deallocate(self%temp)
  allocate(self%idx_rho_aq(0:na),self%idx_gate(0:2))
  do i = 0,2 
    self%idx_gate(i) = basis(2,i)
  end do 
  do i = 0,na 
    self%idx_rho_aq(i) = basis(na,i) 
  end do 
  allocate(self%shiftr(self%size),self%shiftl(self%size),self%shift2r(self%size))
  allocate(self%psi(self%size),self%temp(self%size))
  allocate(self%rho_a(self%dimv,self%dimv),self%rho_aq(self%dimv,self%dimv))
  do s=1,self%size 
    self%shiftr(s) = ishftc(s-1,2,2*lsys)+1
    self%shiftl(s) = ishftc(s-1,-2,2*lsys)+1
    self%shift2r(s) = ishftc(s-1,4,2*lsys)+1
  end do
end function circuit_constructor

! circuit destructor 
subroutine circuit_destructor(self)
  type(circuit) :: self
  if (allocated(self%temp)) deallocate(self%temp)
  if (allocated(self%psi)) deallocate(self%psi)
  if (allocated(self%basis_states)) deallocate(self%basis_states)
  if (allocated(self%state_indices)) deallocate(self%state_indices)
  if (allocated(self%shiftr)) deallocate(self%shiftr)
  if (allocated(self%shiftl)) deallocate(self%shiftl)
  if (allocated(self%shift2r)) deallocate(self%shift2r)
  if (allocated(self%idx_rho_aq)) deallocate(self%idx_rho_aq)
  if (allocated(self%idx_gate)) deallocate(self%idx_gate)
  if (allocated(self%rho_a)) deallocate(self%rho_a)
  if (allocated(self%rho_aq)) deallocate(self%rho_aq)
end subroutine circuit_destructor

! set initial theta state
subroutine set_initial_theta_state(self,theta)
  class(circuit) :: self 
  real(dp) :: theta, m 
  integer :: i, a, sa 
  associate(lsys=>self%lsys, psi=>self%psi,temp=>self%temp,size=>self%size)
    psi = zero
    do i=0,2**lsys-1
      m = zero 
      sa = 0 
      do a=0,lsys-1 
        if (btest(i-1,a)) then 
          m = m+one 
          sa = ibset(sa,2*a)
        end if 
      end do 
      psi(sa+1) = sin(theta/two)**(m)*(cos(theta/two))**(real(lsys,dp)-m)
    end do 
  end associate
end subroutine set_initial_theta_state

subroutine set_initial_theta_neel(self,theta)
  class(circuit) :: self 
  real(dp) :: theta, m1, m2
  integer :: i, a, sa 
  associate(lsys=>self%lsys, psi=>self%psi,temp=>self%temp,size=>self%size)
    psi = zero
    do i=0,2**lsys-1
      m1 = zero 
      m2 = zero
      sa = 0 
      do a=0,lsys-1,2
        if (btest(i-1,a)) then 
          m1 = m1+one 
          sa = ibset(sa,2*a)
        end if 
      end do 
      do a=1,lsys-1,2
        if (btest(i-1,a)) then 
          m2 = m2+one 
          sa = ibset(sa,2*a)
        end if 
      end do 
      psi(sa+1) = cos(theta/two)**(m1)*(-sin(theta/two))**(real(lsys/2,dp)-m1)* &
          & cos(theta/two)**(real(lsys/2,dp)-m2)*(-sin(theta/two))**(m2)
    end do 
  end associate
end subroutine set_initial_theta_neel

! get density matrix 
subroutine get_density_matrix(self)
  class(circuit), intent(in out) :: self 
  complex(dp), allocatable, dimension(:,:) :: red 
  associate(lsys=>self%lsys,psi=>self%psi,rho_a=>self%rho_a,&
    &dimv =>self%dimv, dimc=>self%dimc)
    allocate(red(dimv,dimc))
    red = reshape(psi,(/dimv,dimc/))
    rho_a = matmul(red,transpose(conjg(red)))
    deallocate(red)
  end associate 
end subroutine get_density_matrix

! get projected density matrix 
subroutine get_density_matrix_symproj(self)
  class(circuit), intent(in out) :: self 
  integer :: m 
  associate(rho_a=>self%rho_a,rho_aq=>self%rho_aq, &
    &na=>self%na, base =>self%idx_rho_aq)
    rho_aq = zero 
    do m=0,na 
      rho_aq(base(m)%basis_states+1,base(m)%basis_states+1) = rho_a(base(m)%basis_states+1,base(m)%basis_states+1)
    end do  
  end associate 
end subroutine get_density_matrix_symproj

! get asymmetry 
real(dp) function get_asym(self) result(asym)
  class(circuit), intent(in out) :: self 
  real(dp) :: ent_a, ent_aq
  call self%get_density_matrix()
  call self%get_density_matrix_symproj()
  associate(rho_a =>self%rho_a, rho_aq=>self%rho_aq, dimv =>self%dimv)
    call entanglement_entropy(rho_a,ent_a)
    call entanglement_entropy(rho_aq,ent_aq)
    asym = ent_a - ent_aq 
  end associate
end function get_asym 

! get asymmetry distance
subroutine get_asym_distances(self,asym) 
  class(circuit), intent(in out) :: self 
  real(dp), dimension(2), intent(in out):: asym 
  integer :: s 
  real(dp) :: tr_a, tr_aq
  call self%get_density_matrix()
  call self%get_density_matrix_symproj()
  associate(rho_a =>self%rho_a, rho_aq=>self%rho_aq, dimv =>self%dimv)
    rho_a = matmul(rho_a,rho_a)
    rho_aq = matmul(rho_aq,rho_aq)
    tr_a = zero 
    tr_aq = zero 
    do s = 1,dimv 
      tr_a = tr_a + rho_a(s,s)
      tr_aq = tr_aq + rho_aq(s,s)
    end do 
    asym(1) = tr_a 
    asym(2) = tr_aq 
  end associate
end subroutine get_asym_distances



subroutine generate_gate(self)
  class(circuit) :: self 
  complex(dp), dimension(4,4) :: u0, u2 
  complex(dp), dimension(8,8) :: u1
  call generate_cue(4,u0)
  call generate_cue(8,u1)
  call generate_cue(4,u2)
  associate(gate=>self%gate,base=>self%idx_gate)
    gate = zero 
    gate(base(0)%basis_states+1,base(0)%basis_states+1) = u0 
    gate(base(1)%basis_states+1,base(1)%basis_states+1) = u1 
    gate(base(2)%basis_states+1,base(2)%basis_states+1) = u2 
  end associate 
end subroutine 

! apply U(1) preserving unitary gate on (1,2) spins and shift spins by 2
subroutine apply_u1_gate_and_shift(self)
  class(circuit), intent(in out) :: self 
  integer :: ri, mm, a, ib
  complex(dp) :: a1 
  real(dp) :: r1 
  associate(temp=>self%temp,psi=>self%psi,size=>self%size,gate=>self%gate,t2=>self%shift2r)
    call self%generate_gate()
    mm = size/16
    temp = zero 
    do ri=1,16
      ib = mm*(ri-1)
      do a =1, mm
        temp(ib+a) = sum(gate(ri,:)*psi(a:size:mm))
      end do 
    end do 
    psi(t2) = temp !(t2)
  end associate 
end subroutine apply_u1_gate_and_shift 


! apply U(1) preserving unitary layer 
subroutine apply_u1_unitary_layer(self,it)
  class(circuit), intent(in out) :: self 
  integer, intent(in) :: it
  integer :: u, r, el 
  associate(lsys=>self%lsys,temp=>self%temp,psi=>self%psi,&
    & size=>self%size,t1=>self%shiftr,t2=>self%shift2r,t1r=>self%shiftl)
    if (mod(it,2)==1) psi(t1) = psi
    do el = mod(it,2)+1,lsys-1,2
      call self%apply_u1_gate_and_shift()
    end do 
    if (mod(it,2)==1) psi(t1) = psi
  end associate
end subroutine apply_u1_unitary_layer


end module mod_haar
