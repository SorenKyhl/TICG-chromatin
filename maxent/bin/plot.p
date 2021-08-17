set terminal png
set output 'pconvergence.png'
set key off
p 'convergence.txt' w l

# set output 'pconvergence_diag.png'
# set key off
# p 'convergence_diag.txt' w l

set output 'pchis.png'
set key off
p for [i=1:ARG1] 'chis.txt' u i w l

# set output 'pchis_diag.png'
# set key off
# p for [i=1:ARG2] 'chis_diag.txt' u i w l
