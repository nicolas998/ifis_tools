module read_dat

    contains

    subroutine get_size(path,Ncontrol, Nintervals, Nstates)
        !Define variables
        character*255, intent(in) :: path    
        integer,intent(out) :: Nintervals, Ncontrol, Nstates
        integer i, j
        
        !Open first time just to get the dimensions
        open(unit = 10, file = path, status = 'old', action = 'read')
        read(10,*) Ncontrol
        read(10,*) Nstates
        read(10,*)
        read(10,*) i, Nintervals
        close(10)
    end subroutine

    subroutine get_data(path,Ncontrol, Nintervals, Nstates, control, data)
        !Define variables
        character*255, intent(in) :: path 
        integer, intent(in) ::  Nintervals, Ncontrol, Nstates
        integer,intent(out) :: control(Ncontrol)
        real, intent(out) :: data(Ncontrol,Nstates,Nintervals)
        integer i, j,oe,z
            
        !Open the file to read the data
        open(unit = 10, file = path, status = 'old', action = 'read')
        read(10,*) 
        read(10,*)     
        do i =1,Ncontrol
            read(10,*)
            read(10,*) control(i), oe
            read(10,*) ((data(i,j,z), j = 1,Nstates), z = 1,Nintervals)        
        enddo
        close(10)
    end subroutine

end module