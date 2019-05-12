# Fast Kalman Filter with Sketch

Apply sketch technique on large scale Kalman filter. 
Data sketch method includes ramdom sampling, JL transform, count sketch, adaptive censoring sketch.
Also a sketch projection method is included to speed up generate Kalman gain.
Right now, we are using synthetic data to test those methods's performance.
Some interesting application may come soon.

Check this [notebook](https://github.com/jjkindergarten/data_sketching_kf/blob/master/src/data_sketching_kalman_filter/Data%20Sketching%20Kalman%20Filter%20Comparasion.ipynb)
for the performance of different fast Kalman filter's performance !
Or you may check [this folder](https://github.com/jjkindergarten/data_sketching_kf/tree/master/src/data_sketching_kalman_filter/kalman_filter)
for different performance and running time of different Kalman filter separately.

# Reference
[Berberidis, Dimitris, and Georgios B. Giannakis. "Data sketching for large-scale Kalman filtering." 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.](https://ieeexplore.ieee.org/abstract/document/7472868)

[Gower, Robert M., and Peter Richtárik. "Randomized iterative methods for linear systems." SIAM Journal on Matrix Analysis and Applications 36.4 (2015): 1660-1690.](https://epubs.siam.org/doi/abs/10.1137/15M1025487)