CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Lan Lou
* Tested on:  Windows 10, i7-6700HQ @ 2.6GHz 16GB, GTX 1070 8GB (Personal Laptop)

## Project Features:

- CPU scan: This is the most basic way of doing a prefix sum(looping & adding in cpu).
- Naive scan : what differentiate naive from simple cpu is that this approach happens on gpu, and also we change into using the follwing algorithm shown in the pseudo code:
```for d = 1 to log2n
  for all k in parallel
    if (k >= 2d-1)
      x[k] = x[k – 2d-1] + x[k];
```
which ensure that parallel can be applied.

- Efficient GPU scan : by changing the original naive algorithm into two steps:upsweep and downsweep, we can shorten the amount of operations times it took significantly, the following two shows the pseudo code of up and down sweep:

#### upsweep:
```
for d = 0 to log2n - 1
  for all k = 0 to n – 1 by 2d+1 in parallel
    x[k + 2d+1 – 1] += x[k + 2d – 1];
```

#### downsweep:
```
x[n - 1] = 0
for d = log2n – 1 to 0
  for all k = 0 to n – 1 by 2d+1 in parallel
    t = x[k + 2d – 1];              
    x[k + 2d – 1] = x[k + 2d+1 – 1];  
    x[k + 2d+1 – 1] += t;       
```    
- Efficient scan(optimized)(Extra points) : when finished the former efficient, I discovered that it wasn't even faster than cpu methods, so I went for the optimized version as extra points: the reason for this to happen is that we haven't made full use of all threads in a upsweep or downsweep kernal, since it has braching in it, we will always have some "non-functional" threads,therefore, I decided that instead of judging inside the kernal about if the current loop's thread is a multiple of ```2^(d+1)```, we bring this outside into the scan functions, in it ,we will first  caculate the block num for each loop to be ```blocknum = (adjustlen/interval + blocksize ) / blocksize;``` by dividing the array total length with ```interval = 2^(d-1)```, and also, inside the kernal, we will mutiply the index of threads with ```2^(d-1)``` to make sure it's a multiple of it, as for the others , they will stay the same.

- compact for cpu & efficient: this part algorithm is simmilar for both, as first step, you turn the array into a buffer only containing 1 and 0 through judging if it's 0 or not, then, you do scanning using corresponding method each, after that, we will treat the caculated buffer as outter layer indices to fill into the output buffer, and only taking into acount the inner indices that is having value 1 according to the boolean buffer.

## Questions:
- *Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.*

blocksize/ops' time(ms)|	naive scan(ms)|	efficient scan(ms)
-----|-----|----
128 |	1.911 |	0.541
256 |	1.908 |	0.533
512 |	1.92 |	0.545
1024 |	1.915 |	0.559

- the diagram bellow explicitly shows how variation of block size impact on both the naive and the optimized solutions for scanning, although it's not very apparent, I think block size == 256 is best suited for me, in terms of my own machine condition, algorithms, etc...

![](https://github.com/LanLou123/Project2-Stream-Compaction/raw/master/blocksizechart.JPG)

- *Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).*

Array Size(2^(n))/ops' time(ms)|	Naïve |	Work Efficient(not optimized)|	Work Efficient(optimized) |	Thrust
----|----|----|----|----
3	| 0.032 |	0.029 |	0.093 |	0.085
6	| 0.049	| 0.046	|0.058|	0.065
9	| 0.076	| 0.075	| 0.068	|0.249
12	| 0.09	| 0.137	| 0.135|	0.255
15	| 0.128	| 0.131 |	0.118|	0.263
18	| 0.306	| 0.471 |	0.167	|0.36
21	| 3.89	| 2.311	| 1.008	|0.37
24	| 35.408	| 19.495	|7.925|	0.932
27	| 317.784	| 141.549	|61.03|	5.704

- the corresponding graph is:

![](https://github.com/LanLou123/Project2-Stream-Compaction/raw/master/arraysize.JPG)

- frome the graph we can tell the efficiency of those GPU method arraged in order are: thrust>efficient(optimized)>efficient(not optimized)>naive.

- *Write a brief explanation of the phenomena you see here.*

some performence issues encountered by efficient gpu as i mentioned before is because of invalid use of thread which lead to a lot of wasting, as for the others, one thing that I noticed is that the program will crash when I set the array size to be 2^30,since it's same for all methods, I guess it's the memory IO that's causing this bottleneck.


## Results

### I got this result using the following settings:

- ```blocksize``` = 1024
- ```SIZE``` (in main.cpp) =  1 << 25

```
****************
** SCAN TESTS **
****************
    [  18  25   1  39  31  37   5  10  42  11   4  11  12 ...  37   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 221.761ms    (std::chrono Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 77.1578ms    (std::chrono Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806711 821806713 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 73.3082ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 75.7463ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 15.5116ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 15.3897ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806711 821806713 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.63021ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.63738ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806711 821806713 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   3   1   2   3   0   0   3   1   0   2   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 121.362ms    (std::chrono Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 121.729ms    (std::chrono Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 293.673ms    (std::chrono Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 15.202ms    (CUDA Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   3   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 15.2422ms    (CUDA Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   2   3 ]
    passed
```
