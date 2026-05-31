program main
  use, intrinsic :: iso_fortran_env, dp => real64
  use mod_utils 
  use mod_haar
  use mod_basis
  implicit none 

  integer :: lsys, i, renyi, t, tmax, ndis, nn, cum, na
  integer(dp) :: seed 
  type(circuit) :: b1
  real(dp) :: ent, th, prob, i2, i3, i3odd, i3even, part2, part3, eps
  real(dp), allocatable, dimension(:) :: evq, eva, pur, enta, entaq, entq
  real(dp), dimension(2) :: obs
  logical :: ferro
  character(120), allocatable, dimension(:) :: argv
  character(120) :: fname, ferro_str

  allocate(argv(7))
  do nn = 1, 7
    call get_command_argument(nn, argv(nn))
  end do 
  read(argv(1),*) lsys 
  read(argv(2),*) th 
  read(argv(3),*) seed 
  read(argv(4),*) na
  read(argv(5),*) ndis
  ferro_str = adjustl(argv(6))
  fname = argv(7)
  deallocate(argv)

  if (ferro_str == 'true') then
    ferro = .true.
  else
    ferro = .false.
  end if

  tmax = 15 
  b1 = circuit(lsys, 0.d0, na)
  obs = zero

  allocate(evq(tmax), eva(tmax), pur(tmax), entq(tmax))
  allocate(entaq(tmax), enta(tmax))
  evq = zero
  eva = zero
  pur = zero
  enta = zero
  entaq = zero 
  entq = zero

  do i = 1, ndis
    if (ferro) then
      call b1%set_initial_theta_state(th)
    else
      call b1%set_initial_theta_neel(th)
    end if
    do t = 1,10 !tmax
      call b1%get_asym_distances(obs)
      evq(t) = evq(t) + (one - obs(2)/obs(1))/real(ndis, dp)
      eva(t) = eva(t) + (obs(2))/real(ndis, dp)
      pur(t) = pur(t) + (obs(1))/real(ndis, dp)
      entq(t) = entq(t) + log(obs(2)/obs(1))/real(ndis, dp)
      entaq(t) = entaq(t) + log(obs(2))/real(ndis, dp)
      enta(t) = enta(t) + log(obs(1))/real(ndis, dp)
      call b1%apply_u1_unitary_layer(1)
      call b1%apply_u1_unitary_layer(2)
    end do 
  end do 

  do t = 1, tmax 
    open(unit=66, file=fname, action='write', position='append')
    write(66, '(I8,6(F15.6))') t, evq(t), eva(t), pur(t), entq(t), entaq(t), enta(t)
    close(66)
  end do   

  deallocate(evq, eva, pur, enta, entaq, entq) 

end program main