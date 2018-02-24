# Standalone-and-portable-Cpp11-matrix-library
Standalone and portable C++11 matrix library

There are several good C++ matrix libraries that could be used for Computer Vision, Machine Learning and AI applications. However, most of them require complex compilation, dependencies with third party components such as Fortran libraries, etc. Moreover, it is not convenient to simply include most of these existing libraries simply by including a single header file. Furthermore, the majority of the existing libraries do not use modern C++ (e.g. C++11) features.

The project provides a set of matrix and vector classes that do not depend on any external libraries; only just standard C++11 libraries are required. The internal matrix storage is the column major scheme. The user can very easily interface and convert back/forth from/to any other matrix libraries and std::vector<T> objects. The classes allow performing many of the matrix operations that are available in MATLAB.

Similar to MATLAB, every operation is a deep copy using the very fast "memcpy", which is afforded by the fact that the entire matrix is stored contiguously. This means that the users do not have to worry about whether some operation will accidentally change the matrices which may happen in other libraries such as Armadillo, Eigen or Opencv.

Moreover, matrix classes given in libraries such as OpenCV can be inconsistent in many places. For example, OpenCV matrix operations sometimes create references and sometimes create deep copies. This makes it hard to remember, keep track of and make the users be less confident of any new code they are writing. Moreover, they could also change the nature of these operations in their library in the future. The users can certainly keep their old versions, but due to the complex and inconsistent nature of the code, they may not able to easily modify/adapt it.

Furthermore, even the matrix types are not well documented and filled with things like CV_8UC3 and Vec3b. In fact, the whole idea can be unified and made much simpler. The matrix classes in this project allow 2D matrices with any number of channels (i.e. either 2D or 3D matrices). Moreover, every matrix is always guaranteed to be stored in a contiguous manner, which makes it very cache friendly (and therefore fast) and convenient to interface with other libraries.

Each element of the matrix can be any native C++ type (such as int and float). Some of the standard headers used in the library are <iostream>, <algorithm>, <vector>, <numeric>, <random>, <cstdio>, <cmath>, <cstring>, <cstdlib>, <cstdint>. These matrix classes make good use of C++ standard libraries and C++11 features where relevant. These classes could therefore, to a certain extent, fills the need for a matrix class in the C++ standard library. This matrix classes, by design, do not make use of any C++ STL-style iterators etc., to make the entire code as efficient, clear, simple and intuitive as possible. The core structure is a dynamic array handled by a smart pointer "unique_ptr" which has no overhead. This class also makes extensive use of efficient C++ standard library algorithms such as std::max, min, transform, sort, nth_element, accumulate, lower_bound, distance, various random number generators and lamda functions.

Some examples of the usage  of the library is given below, however for most of the operations not shown here, kindly refer directly to the code as I currently do not have the time to document everything thoroughly.

https://kyaw.xyz/2017/12/18/standalone-portable-cpp11-matrix-library

Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

Dr. Kyaw Kyaw Htike @ Ali Abdul Ghafur

https://kyaw.xyz


/*

Some library usage examples (which only cover a very small of operations; for others, refer to the library directly):

// create a matrix using initializer list

Matkr<float> a(4, 5, 2, { 52,37,74,53,81,82,19,13,83,64,2,90,52,55,61,77,86,39,9,74,34,84,38,83,18,13,88,5,69,74,44,38,98,40,45,16,33,32,90,25 });

// print the matrix

cout << a << endl;

// remove columns 1 and 3

a = a.rem_cols({1,3});
cout << a << endl;

// get channel zero of the matrix a, add rows to the matrix from
// the previous matrix a's 1st and 2nd rows

a = a.gch(0).add_rows(a.gr(0, 1));
cout << a << endl;

Output:

Channel 1:
52.0000 81.0000 83.0000 52.0000 86.0000
37.0000 82.0000 64.0000 55.0000 39.0000
74.0000 19.0000 2.0000 61.0000 9.0000
53.0000 13.0000 90.0000 77.0000 74.0000
=========================
Channel 2:
34.0000 18.0000 69.0000 98.0000 33.0000
84.0000 13.0000 74.0000 40.0000 32.0000
38.0000 88.0000 44.0000 45.0000 90.0000
83.0000 5.0000 38.0000 16.0000 25.0000
=========================

Channel 1:
52.0000 83.0000 86.0000
37.0000 64.0000 39.0000
74.0000 2.0000 9.0000
53.0000 90.0000 74.0000
=========================
Channel 2:
34.0000 69.0000 33.0000
84.0000 74.0000 32.0000
38.0000 44.0000 90.0000
83.0000 38.0000 25.0000
=========================

Channel 1:
52.0000 83.0000 86.0000
37.0000 64.0000 39.0000
74.0000 2.0000 9.0000
53.0000 90.0000 74.0000
52.0000 83.0000 86.0000
37.0000 64.0000 39.0000
=========================
Channel 2:
0.0000 0.0000 0.0000
0.0000 0.0000 0.0000
0.0000 0.0000 0.0000
0.0000 0.0000 0.0000
34.0000 69.0000 33.0000
84.0000 74.0000 32.0000
=========================

Press any key to continue . . .

Code (cont'd):

// get the channel zero of the matrix a

Matkrf b = a.gch(0);
cout << b << endl;

Output:

Channel 1:
52.0000 83.0000 86.0000
37.0000 64.0000 39.0000
74.0000 2.0000 9.0000
53.0000 90.0000 74.0000
=========================

Press any key to continue . . .

Code (cont'd):

// get the submatrix corresponding to
// row 1 and 3, and col 1 and 2

cout << b.gs({ 1,3 }, { 1,2 }) << endl;

Output:

Channel 1:
64.0000 39.0000
90.0000 74.0000

Code (cont'd):

// get the matrix from continuous range
// row 1 to 3, and col 1 to 2

cout << b.gs(1,3,1,2) << endl;

Output:

Channel 1:
64.0000 39.0000
2.0000 9.0000
90.0000 74.0000

Code (cont'd):

// replace the matrix b's row1 to row 3 and col
// 1 to col 2 with a matrix that is randomly
// uniformly sampled integers between -10 and -5.

b.ss(1, 3, 1, 2) = Matkrf::randi(3, 2, 1, -10, -5);
cout << b << endl;

Output:

Channel 1:
52.0000 83.0000 86.0000
37.0000 -9.0000 -7.0000
74.0000 -5.0000 -7.0000
53.0000 -7.0000 -9.0000

Code:

Matkrd d = { 99.5, 45, 3, 40 };
cout << d << endl;

Output:

Channel 1:
99.5000
45.0000
3.0000
40.0000

Code:

Matkrd d({ 99.5, 45, 3, 40 });
cout << d << endl;

Output:

Channel 1:
99.5000
45.0000
3.0000
40.0000

Code:

// make row vector
Matkrd d({ 99.5, 45, 3, 40 }, false);
cout << d << endl;

Output:

Channel 1:
99.5000 45.0000 3.0000 40.0000

Code:

Matkr<TT> a(4, 5, 2, { 52,37,74,53,81,82,19,13,83,64,2,90,52,55,61,77,86,39,9,74,34,84,38,83,18,13,88,5,69,74,44,38,98,40,45,16,33,32,90,25 });

cout << "before: " << endl;
cout << a << endl;
cout << "after: " << endl;
a.ss({ 0,1 }, { 2,3 }, { 0 }) = Matkr<TT>(2, 2, 1, { 1,2,3,4 });
cout << a << endl;

Output:

before:

Channel 1:
52.0000 81.0000 83.0000 52.0000 86.0000
37.0000 82.0000 64.0000 55.0000 39.0000
74.0000 19.0000 2.0000 61.0000 9.0000
53.0000 13.0000 90.0000 77.0000 74.0000
=========================
Channel 2:
34.0000 18.0000 69.0000 98.0000 33.0000
84.0000 13.0000 74.0000 40.0000 32.0000
38.0000 88.0000 44.0000 45.0000 90.0000
83.0000 5.0000 38.0000 16.0000 25.0000
=========================

after:

Channel 1:
52.0000 81.0000 1.0000 3.0000 86.0000
37.0000 82.0000 2.0000 4.0000 39.0000
74.0000 19.0000 2.0000 61.0000 9.0000
53.0000 13.0000 90.0000 77.0000 74.0000
=========================
Channel 2:
34.0000 18.0000 69.0000 98.0000 33.0000
84.0000 13.0000 74.0000 40.0000 32.0000
38.0000 88.0000 44.0000 45.0000 90.0000
83.0000 5.0000 38.0000 16.0000 25.0000
=========================

*/
