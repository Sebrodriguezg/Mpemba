program main
  use, intrinsic :: iso_fortran_env, dp => real64
  use mod_utils 
  use mod_haar
  use mod_basis
  implicit none 

  integer :: lsys, t, tmax, na, ndis, nn
  integer(dp) :: seed 
  type(circuit) :: b1
  real(dp) :: th
  real(dp), dimension(2) :: obs, ent
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

  if (ferro_str == 'true') then
    ferro = .true.
  else
    ferro = .false.
  end if
  deallocate(argv)

  call init_random_seed(seed)

  tmax = 40
  b1 = circuit(lsys, na)

  do nn = 1, ndis 
    if (ferro) then
      call b1%set_initial_theta_state(th)
    else
      call b1%set_initial_theta_neel(th)
    end if

    do t = 0, tmax
      call b1%get_asymmetry_distances(obs)
      call b1%get_entanglement_asymmetry(ent)

      
      call b1%apply_u1_unitary_layer(1)
      call b1%apply_u1_unitary_layer(2)

      open(unit=66, file=fname, action='write', position='append')
      write(66, '(4(I8), 5(F15.8))') nn, lsys, na, t, th, obs, ent
      close(66)
    end do 
  end do 

end program main