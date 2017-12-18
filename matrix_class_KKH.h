#ifndef MATRIX_CLASS_KKH_H
#define MATRIX_CLASS_KKH_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

// forward declarations
template <class T>
class Matk;
template <class T>
class Matk_ViewRef;
template <class T>
class Matkr;
template <class T>
class Matkr_ViewRef;
template <class T>
class Veck;
template <class T>
class Veck_ViewRef;

template <class T>
class Matk
{
public:
	
	Matk() {}
	
	~Matk() {}
	
	Matk(int nrows, int ncols, int nchannels)
	{
		init(nrows, ncols, nchannels);
	}
	
	Matk(int nrows, int ncols)
	{
		init(nrows, ncols, 1);
	}

	Matk(int nrows, int ncols, int nchannels, const T* data_external, bool external_is_row_major=false)
	{
		// data from outside is always copied
		init(nrows, ncols, nchannels, data_external, external_is_row_major);
	}

	Matk(int nrows, int ncols, int nchannels, std::initializer_list<T> iList)
	{
		std::vector<T> v(iList);
		// data from outside is always copied
		init(nrows, ncols, nchannels, v.data());
	}

	// creates a column vector
	Matk(std::initializer_list<T> iList, bool column_vec=true)
	{
		if (column_vec)
			init(iList.size(), 1, 1, iList.begin());
		else
			init(1, iList.size(), 1, iList.begin());
	}

	// row/column vector from std::vector (deep copy)
	Matk(const std::vector<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.data()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.data()); // create row vector
	}

	// row/column vector from Veck vector (deep copy)
	Matk(const Veck<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.get_ptr()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.get_ptr()); // create row vector
	}

	// matrix from std::vector (deep copy)
	Matk(int nrows, int ncols, int nchannels, const std::vector<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.data());
		}
		else
			init(nrows, ncols, nchannels, vecIn.data());
	}

	Matk(int nrows, int ncols, int nchannels, const Veck<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.get_ptr());
		}
		else
			init(nrows, ncols, nchannels, vecIn.get_ptr());
	}

	// copy constructor (deep copy); e.g. Matk<T> m3 = m1;  //same as "Matk<T> m3(m1);"
	// where m1 is a previously allocated/used matrix
	Matk(const Matk<T> &matIn)
	{
		init(matIn.nrows(), matIn.ncols(), matIn.nchannels(), matIn.get_ptr());
	}
	
	Matk(const Matk_ViewRef<T> &matIn)
	{
		create(matIn);
	}

	void create(int nrows, int ncols, int nchannels)
	{
		init(nrows, ncols, nchannels);
	}
	
	void create(int nrows, int ncols)
	{
		init(nrows, ncols, 1);
	}

	void create(int nrows, int ncols, int nchannels, const T* data_external)
	{
		// data from outside is always copied
		init(nrows, ncols, nchannels, data_external);
	}

	// create matrix from std::vector (deep copy)
	void create(int nrows, int ncols, int nchannels, const std::vector<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.data());
		}
		else
			init(nrows, ncols, nchannels, vecIn.data());
	}

	// create matrix from Veck vector (deep copy)
	void create(int nrows, int ncols, int nchannels, const Veck<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.get_ptr());
		}
		else
			init(nrows, ncols, nchannels, vecIn.get_ptr());
	}
	
	// create row/column vector from std::vector (deep copy)
	void create(const std::vector<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.data()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.data()); // create row vector
	}

	void create(const Veck<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.get_ptr()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.get_ptr()); // create row vector
	}
	
	void create(const Matk_ViewRef<T> &matIn)
	{
		int nr_in = matIn.mm.nrows();
		int nc_in = matIn.mm.ncols();
		int nch_in = matIn.mm.nchannels();

		Matk<T> &mm = matIn.mm;
		T* ptr_in = matIn.mm.get_ptr();

		if (matIn.cont_range)
		{
			int r1 = matIn.r1;
			int r2 = matIn.r2;
			int c1 = matIn.c1;
			int c2 = matIn.c2;
			int ch1 = matIn.ch1;
			int ch2 = matIn.ch2;

			int nch_s = ch2 - ch1 + 1;
			int nc_s = c2 - c1 + 1;
			int nr_s = r2 - r1 + 1;

			init(nr_s, nc_s, nch_s);

			// special case for when all the channels & all rows are included
			// in matlab notation, this is like mat(:,c1:c2,:)
			// this is the fastest case since the underlying is stored in
			// col major order and I can simply use a single memcpy.
			if (nch_s == nch_in && nr_s == nr_in)
			{
				std::memcpy(ptr, ptr_in + c1*nch_in*nr_in, sizeof(T)*(nch_in*nr_s));
			}

			// special case for when all the channels are included but all not rows.
			// in matlab notation, this is like mat(r1:r2,c1:c2,:).
			// this is not as fast as the first case but still much faster than the
			// general case below.
			else if (nch_s == nch_in && nr_s != nr_in)
			{
				for (unsigned int j = c1; j <= c2; j++)
				{
					std::memcpy(ptr + (j - c1)*nch_in*nr_s,
						ptr_in + j*nch_in*nr_in + r1*nch_in,
						sizeof(T)*(nch_in*nr_s));
				}
			}

			// general case and will be rthe slowest.
			// in matlab notation, this is like mat(r1:r2,c1:c2,ch1:ch2).
			else
			{
				unsigned int cc = 0;						
				for (unsigned int k = ch1; k <= ch2; k++)
					for (unsigned int j = c1; j <= c2; j++)
						for (unsigned int i = r1; i <= r2; i++)	
							mm(i, j, k) = ptr_in[cc++];
			}

		}

		else
		{
			Veck<int>row_indices = matIn.row_indices;
			Veck<int>col_indices = matIn.col_indices;
			Veck<int>ch_indices = matIn.ch_indices;

			// if any range is given in the form specified in the comment above,
			// transform these range values to indices (continuous numbers)
			if ((row_indices.size() == 2) && (row_indices[0] <= 0) && (row_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((row_indices[0] == 0) && (row_indices[1] == -1)) { start_idx = 0; end_idx = nr_in - 1; }
				else { start_idx = abs(row_indices[0]); end_idx = abs(row_indices[1]); }
				//row_indices.resize(end_idx - start_idx + 1);
				//std::iota(row_indices.begin(), row_indices.end(), start_idx);
				row_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((col_indices.size() == 2) && (col_indices[0] <= 0) && (col_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((col_indices[0] == 0) && (col_indices[1] == -1)) { start_idx = 0; end_idx = nc_in - 1; }
				else { start_idx = abs(col_indices[0]); end_idx = abs(col_indices[1]); }
				//col_indices.resize(end_idx - start_idx + 1);
				//std::iota(col_indices.begin(), col_indices.end(), start_idx);
				col_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((ch_indices.size() == 2) && (ch_indices[0] <= 0) && (ch_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((ch_indices[0] == 0) && (ch_indices[1] == -1)) { start_idx = 0; end_idx = nch_in - 1; }
				else { start_idx = abs(ch_indices[0]); end_idx = abs(ch_indices[1]); }
				//ch_indices.resize(end_idx - start_idx + 1);
				//std::iota(ch_indices.begin(), ch_indices.end(), start_idx);
				ch_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}

			int rsize = row_indices.size();
			int csize = col_indices.size();
			int chsize = ch_indices.size();

			init(rsize, csize, chsize);

			unsigned int cc = 0;
			for (unsigned int k = 0; k < chsize; k++)
				for (unsigned int j = 0; j < csize; j++)
					for (unsigned int i = 0; i < rsize; i++)
						ptr[cc++] = mm(row_indices[i], col_indices[j], ch_indices[k]);
		}
	}
		
	// wrap an external contiguous array stored in column major order.
	// this class will not manage that external memory, it's the 
	// responsibility of that external source to delete that memory.
	// When wrapping, the class will release its current dynamic
	// being managed (if any) by the unique_ptr before taking on this external
	// pointer. This method is useful if I don't want to make a copy of
	// that external memory and simply use the methods of that class on it.
	// This is great for many purposes such as when the external data is huge
	// and I want to simply do some operations on it or get slices from it
	// etc.
	void wrap(int nrows, int ncols, int nchannels, T* data_external)
	{
		data.reset(); // free currently owned memory by the class
		nr = nrows; nc = ncols; nch = nchannels;
		ndata = nr * nc * nch;
		ptr = data_external; // no managing of data_external, simply a pointer pointing towards it.
	}

	// convert this matrix to std::vector
	std::vector<T> toVec()
	{
		std::vector<T> v(ndata);
		std::memcpy(v.data(), ptr, sizeof(T)*ndata);
		return v;
	}

	// convert this matrix to Veck vector
	std::vector<T> toVeck()
	{
		Veck<T> v(ndata);
		std::memcpy(v.get_ptr(), ptr, sizeof(T)*ndata);
		return v;
	}
	
	// assignment operator (deep copy); e.g. m3 = m1; // where m1 is a previously allocated/used matrix
	void operator=(const Matk<T> &matIn)
	{
		init(matIn.nrows(), matIn.ncols(), matIn.nchannels(), matIn.get_ptr());
	}

	// create a column vector from std::vector (deep copy)
	void operator=(const std::vector<T> vecIn)
	{
		init(vecIn.size(), 1, 1, vecIn.data()); // create column vector
	}

	// create a column vector from Veck vector (deep copy)
	void operator=(const Veck<T> vecIn)
	{
		init(vecIn.size(), 1, 1, vecIn.get_ptr()); 
	}
	
	void operator=(const Matk_ViewRef<T> &matIn)
	{
		create(matIn);
	}

	// individual element access (read)
	T operator()(int i, int j, int k) const
	{
		return ptr[k*nr*nc + j*nr + i];
	}

	// get submatrix by indices (read)
	Matk<T> operator()(Veck<int> row_indices, Veck<int> col_indices, Veck<int> ch_indices) const
	{
		return gs(row_indices, col_indices, ch_indices);
	}

	// get submatrix by indices (read)
	Matk<T> operator()(Veck<int> row_indices, Veck<int> col_indices) const
	{
		return gs(row_indices, col_indices);
	}

	// individual element access (read) (assumes k=0, i.e. referring to 1st channel)
	T operator()(int i, int j) const
	{
		return ptr[j*nr + i];
	}

	// individual element access (write)
	T& operator()(int i, int j, int k)
	{
		return ptr[k*nr*nc + j*nr + i];
	}

	// individual element access (write)  (assumes k=0, i.e. referring to 1st channel)
	T& operator()(int i, int j)
	{
		return ptr[j*nr + i];
	}
	
	// individual element access (read)
	T operator[](int linear_index) const
	{
		return ptr[linear_index];
	}

	// individual element access (write)
	T& operator[](int linear_index)
	{
		return ptr[linear_index];
	}	

	void operator++(int)
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]++;
	}

	void operator++()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]++;
	}

	void operator--(int)
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]--;
	}

	void operator--()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]--;
	}

	// check whether another matrix is approximately equal to this matrix
	bool operator== (const Matk<T> &matIn)
	{
			if ((matIn.nrows() != nr) || (matIn.ncols() != nc) || (matIn.nchannels() != nch))
				return false;		

			T* ptr_matIn = matIn.get_ptr();

			// check every element of matrix
			// don't check for exact equality, approximate equality with some threshold
			for (int i = 0; i < ndata; i++)
			{
				if (  abs( double(ptr_matIn[i]) - double(ptr[i]) ) > 0.00000001  )
					return false;
			}

			return true;
	}

	// check whether another matrix is NOT approximately equal to this matrix
	bool operator!= (const Matk<T> &matIn)
	{
		return !((*this) == matIn);
	}

	// element-wise multiplication two matrices
	Matk<T> operator% (const Matk<T> &matIn)
	{
		Matk<T> matOut(nr, nc, nch);

		T* ptr_matIn = matIn.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn.ndata; i++)
			ptr_matOut[i] = ptr[i] * ptr_matIn[i];

		return matOut;
	}
	
	// add two matrices
	friend Matk<T> operator+ (const Matk<T> &matIn1, const Matk<T> &matIn2) 
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matIn2 = matIn2.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] + ptr_matIn2[i];

		return matOut;
	}

	// add a matrix and a number
	friend Matk<T> operator+ (const Matk<T> &matIn1, T sc)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] + sc;

		return matOut;
	}

	// add a number and a matrix
	friend Matk<T> operator+ (T sc, const Matk<T> &matIn1)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc + ptr_matIn1[i];

		return matOut;
	}

	// minus two matrices
	friend Matk<T> operator- (const Matk<T> &matIn1, const Matk<T> &matIn2)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matIn2 = matIn2.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] - ptr_matIn2[i];

		return matOut;
	}

	// minus a matrix and a number
	friend Matk<T> operator- (const Matk<T> &matIn1, T sc)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] - sc;

		return matOut;
	}

	// minus a number and a matrix
	friend Matk<T> operator- (T sc, const Matk<T> &matIn1)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc - ptr_matIn1[i];

		return matOut;
	}

	// element-wise divide two matrices
	friend Matk<T> operator/ (const Matk<T> &matIn1, const Matk<T> &matIn2)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matIn2 = matIn2.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] / ptr_matIn2[i];

		return matOut;
	}

	// divide a matrix by a scalar
	friend Matk<T> operator/ (const Matk<T> &matIn1, T sc)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();		
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] / sc;

		return matOut;
	}

	// divide a scalar by a matrix
	friend Matk<T> operator/ (T sc, const Matk<T> &matIn1)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc / ptr_matIn1[i];

		return matOut;
	}

	// matrix multiplication of two matrices
	// assume nchannels == 1 for both matrices
	friend Matk<T> operator* (const Matk<T> &matIn1, const Matk<T> &matIn2)
	{
		int nr_new = matIn1.nr;
		int nc_new = matIn2.nc;

		Matk<T> matOut(nr_new, nc_new, 1);

		T* ptr_out = matOut.get_ptr();

		unsigned int c = 0;

		for (int j = 0; j < nc_new; j++)
			for (int i = 0; i < nr_new; i++)
				ptr_out[c++] = matIn1.gr(i).dot(matIn2.gc(j));				

		return matOut;
	}

	// multiply a matrix and a scalar value
	friend Matk<T> operator* (const Matk<T> &matIn1, T sc)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] * sc;

		return matOut;
	}

	// multiply a scalar value and a matrix
	friend Matk<T> operator* (T sc, const Matk<T> &matIn1)
	{
		Matk<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc * ptr_matIn1[i];

		return matOut;
	}

	friend std::ostream& operator<<(std::ostream &os, const Matk<T> &matIn)
	{
		for (unsigned int k = 0; k < matIn.nchannels(); k++)
		{
			os << "Channel " << k + 1 << ":" << endl;
			for (unsigned int i = 0; i < matIn.nrows(); i++)
			{
				for (unsigned int j = 0; j < matIn.ncols(); j++)
				{
					os << std::setprecision(4) << std::fixed << matIn(i, j, k) << "\t";
				}
				os << endl;
			}
			os << "=========================" << endl;
		}
		return os;
	}

	// get submatrix from given range
	Matk<T> gs(int r1, int r2, int c1, int c2, int ch1, int ch2) const
	{
		if (r1 == -1) r1 = nr - 1; // -1 is a special notation for last one
		if (r2 == -1) r2 = nr - 1; // -1 is a special notation for last one
		if (c1 == -1) c1 = nc - 1; // -1 is a special notation for last one
		if (c2 == -1) c2 = nc - 1; // -1 is a special notation for last one
		if (ch1 == -1) ch1 = nch - 1; // -1 is a special notation for last one
		if (ch2 == -1) ch2 = nch - 1; // -1 is a special notation for last one

		int nch_s = ch2 - ch1 + 1;
		int nc_s = c2 - c1 + 1;
		int nr_s = r2 - r1 + 1;

		Matkr<T> matOut(nr_s, nc_s, nch_s);
		T* ptr_out = matOut.get_ptr();		
		
		// special case for when all the channels & all rows are included
		// in matlab notation, this is like mat(:,c1:c2,:)
		// this is the fastest case since the underlying is stored in
		// col major order and I can simply use a single memcpy.
		if(nch_s==nch && nr_s==nr) 
		{		
			std::memcpy(ptr_out, ptr+c1*nch*nr, sizeof(T)*(nch*nr_s));
		}
		
		// special case for when all the channels are included but all not rows.
		// in matlab notation, this is like mat(r1:r2,c1:c2,:).
		// this is not as fast as the first case but still much faster than the
		// general case below.
		else if (nch_s==nch && nr_s!=nr) 
		{
			for (unsigned int j = c1; j <= c2; j++)
			{
				std::memcpy(ptr_out+(j-c1)*nch*nr_s, ptr+j*nch*nr+r1*nch, sizeof(T)*(nch*nr_s));
			}
		}
		
		// general case and will be the slowest.
		// in matlab notation, this is like mat(r1:r2,c1:c2,ch1:ch2).
		else
		{
			unsigned int cc = 0;
			for (unsigned int k = ch1; k <= ch2; k++)
				for (unsigned int j = c1; j <= c2; j++)
					for (unsigned int i = r1; i <= r2; i++)
						ptr_out[cc++] = (*this)(i, j, k);								
		}
				
		return matOut;
	}
	
	// get submatrix given indices & range combination
	// apart from vector of index values,
	// optionally allow range specification which is denoted by two-value
	// vector, both of which are <= zero. 
	// e.g. {-3,-7} means continuous range from 3 to 7.
	// e.g. {0,10} means continous range from 0 to 10.
	// a special case is {0,-1} which denotes entire range, i.e. 0 to nrows/ncols/nch
	// does deep copy
	Matk<T> gs(Veck<int> row_indices, Veck<int> col_indices, Veck<int> ch_indices) const
	{

		// if any range is given in the form specified in the comment above,
		// transform these range values to indices (continuous numbers)
		if ((row_indices.size()==2) && (row_indices[0] <= 0) && (row_indices[1] <= 0))
		{
			int start_idx, end_idx;
			if ((row_indices[0] == 0) && (row_indices[1] == -1)) { start_idx = 0; end_idx = nr - 1; }
			else { start_idx = abs(row_indices[0]); end_idx = abs(row_indices[1]); }				
			//row_indices.resize(end_idx - start_idx + 1);
			//std::iota(row_indices.begin(), row_indices.end(), start_idx); 
			row_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
		}
		if ((col_indices.size() == 2) && (col_indices[0] <= 0) && (col_indices[1] <= 0))
		{
			int start_idx, end_idx;
			if ((col_indices[0] == 0) && (col_indices[1] == -1)) { start_idx = 0; end_idx = nc - 1; }
			else { start_idx = abs(col_indices[0]); end_idx = abs(col_indices[1]); }				
			//col_indices.resize(end_idx - start_idx + 1);
			//std::iota(col_indices.begin(), col_indices.end(), start_idx);
			col_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
		}
		if ((ch_indices.size() == 2) && (ch_indices[0] <= 0) && (ch_indices[1] <= 0))
		{
			int start_idx, end_idx;
			if ((ch_indices[0] == 0) && (ch_indices[1] == -1)) { start_idx = 0; end_idx = nch - 1; }
			else { start_idx = abs(ch_indices[0]); end_idx = abs(ch_indices[1]); }				
			//ch_indices.resize(end_idx - start_idx + 1);
			//std::iota(ch_indices.begin(), ch_indices.end(), start_idx);
			ch_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
		}
		
		int rsize = row_indices.size();
		int csize = col_indices.size();
		int chsize = ch_indices.size();

		Matk<T>matOut(rsize, csize, chsize);

		T* ptr_out = matOut.get_ptr();
		unsigned int cc = 0;

		for (unsigned int k = 0; k < chsize; k++)
			for (unsigned int j = 0; j < csize; j++)
				for (unsigned int i = 0; i < rsize; i++)
					ptr_out[cc++] = (*this)(row_indices[i], col_indices[j], ch_indices[k]);

		return matOut;
	}
	
	// get submatrix from given range (assumes all channels requested)
	Matk<T> gs(int r1, int r2, int c1, int c2) const
	{
		return gs(r1, r2, c1, c2, 0, -1);
	}	
	
	// assume all channels are requested
	Matk<T> gs(Veck<int> row_indices, Veck<int> col_indices) const
	{
		return gs(row_indices, col_indices, {0, -1});
	}

	// get specific columns of matrix from range c1:c2
	// assumes all rows and channels are requested
	Matk<T> gc(int c1, int c2) const
	{
		return gs(0, -1, c1, c2, 0, -1);
	}

	// get specific columns of matrix from given indices
	// assumes all rows and channels are requested
	Matk<T> gc(Veck<int> col_indices) const
	{
		return gs({ 0, -1 }, col_indices, { 0, -1 });
	}

	// get a specific column of matrix
	// assumes all rows and channels are requested
	Matk<T> gc(int c) const
	{
		return gs(0, -1, c, c, 0, -1);
	}

	// get specific rows of matrix from range r1:r2
	// assumes all cols and channels are requested
	Matk<T> gr(int r1, int r2)  const
	{
		return gs(r1, r2, 0, -1, 0, -1);
	}

	// get specific rows of matrix from given indices
	// assumes all cols and channels are requested
	Matk<T> gr(Veck<int> row_indices) const
	{
		return gs(row_indices, { 0, -1 }, { 0, -1 });
	}

	// get a specific row of matrix
	// assumes all cols and channels are requested
	Matk<T> gr(int r) const
	{
		return gs(r, r, 0, -1, 0, -1);
	}

	// get specific channels of matrix from range ch1:ch2
	// assumes all rows and cols are requested
	Matk<T> gch(int ch1, int ch2) const
	{
		return gs(0, -1, 0, -1, ch1, ch2);
	}

	// get specific channels of matrix from given indices
	// assumes all rows and cols are requested
	Matk<T> gch(Veck<int> ch_indices) const
	{
		return gs({ 0, -1 }, { 0, -1 }, ch_indices);
	}

	// get a specific channel of matrix
	// assumes all rows and cols are requested
	Matk<T> gch(int ch) const
	{
		return gs(0, -1, 0, -1, ch, ch);
	}

	// get ref to submatrix from given range
	// can also be used as LHS to set values in the matrix
	Matk_ViewRef<T> ss(int r1, int r2, int c1, int c2, int ch1, int ch2)
	{
		if (r1 == -1) r1 = nr - 1; // -1 is a special notation for last one
		if (r2 == -1) r2 = nr - 1; // -1 is a special notation for last one
		if (c1 == -1) c1 = nc - 1; // -1 is a special notation for last one
		if (c2 == -1) c2 = nc - 1; // -1 is a special notation for last one
		if (ch1 == -1) ch1 = nch - 1; // -1 is a special notation for last one
		if (ch2 == -1) ch2 = nch - 1; // -1 is a special notation for last one
		
		return Matk_ViewRef<T>(*this, r1, r2, c1, c2, ch1, ch2);
	}

	// get ref to submatrix from given range 
	// can also be used as LHS to set values in the matrix
	// assumes all channels are requested
	Matk_ViewRef<T> ss(int r1, int r2, int c1, int c2)
	{
		return ss(r1, r2, c1, c2, 0, -1);
	}
	
	// get ref to submatrix given indices & range combination.
	// can also be used as LHS to set values in the matrix
	Matk_ViewRef<T> ss(Veck<int> row_indices, Veck<int> col_indices, Veck<int> ch_indices)
	{		
		return Matk_ViewRef<T>(*this, row_indices, col_indices, ch_indices);
	}

	// get ref to submatrix given indices & range combination
	// can also be used as LHS to set values in the matrix
	// assumes all channels are requested
	Matk_ViewRef<T> ss(Veck<int> row_indices, Veck<int> col_indices)
	{
		return ss(row_indices, col_indices, { 0, -1 });
	}
	
	// get ref to specific columns of matrix from range c1:c2 
	// can also be used as LHS to set values in the matrix
	// assumes all rows and channels are requested
	Matk_ViewRef<T> sc(int c1, int c2)
	{
		return ss(0, -1, c1, c2, 0, -1);
	}

	// get ref to set specific columns of matrix from column indices
	// can also be used as LHS to set values in the matrix
	// assumes all rows and channels are requested
	Matk_ViewRef<T> sc(Veck<int> col_indices)
	{
		return ss({ 0, -1 }, col_indices, { 0, -1 });
	}

	// get ref to a specific column of matrix
	// can also be used as LHS to set values in the matrix
	// assumes all rows and channels are requested
	Matk_ViewRef<T> sc(int c)
	{
		return ss(0, -1, c, c, 0, -1);
	}

	// get ref to specific rows of matrix from range r1:r2
	// can also be used as LHS to set values in the matrix
	// assumes all cols and channels are requested
	Matk_ViewRef<T> sr(int r1, int r2)
	{
		return ss(r1, r2, 0, -1, 0, -1);
	}

	// get ref to specific rows of matrix from row indices
	// can also be used as LHS to set values in the matrix
	// assumes all cols and channels are requested
	Matk_ViewRef<T> sr(Veck<int> row_indices)
	{
		return ss(row_indices, { 0, -1 }, { 0, -1 });
	}
	
	// get ref to a specific row of matrix
	// can also be used as LHS to set values in the matrix
	// assumes all cols and channels are requested
	Matk_ViewRef<T> sr(int r)
	{
		return ss(r, r, 0, -1, 0, -1);
	}

	// get ref to specific channels of matrix from range ch1:ch2
	// can also be used as LHS to set values in the matrix
	// assumes all rows and cols are requested
	Matk_ViewRef<T> sch(int ch1, int ch2)
	{
		return ss(0, -1, 0, -1, ch1, ch2);
	}

	// get ref to specific columns of matrix from channel indices
	// can also be used as LHS to set values in the matrix
	// assumes all rows and cols are requested
	Matk_ViewRef<T> sch(Veck<int> ch_indices)
	{
		return ss({ 0, -1 }, { 0, -1 }, ch_indices);
	}

	// get ref to a specific channel of matrix
	// can also be used as LHS to set values in the matrix
	// assumes all rows and cols are requested
	Matk_ViewRef<T> sch(int ch)
	{
		return ss(0, -1, 0, -1, ch, ch);
	}

	// get a column/row vector out of the matrix by flattening
	Matk<T> vectorize(bool col_vec = true) const
	{
		if (col_vec)
			return Matk<T>(ndata, 1, 1, ptr);
		else
			return Matk<T>(1, ndata, 1, ptr);
	}

	// perform dot product between two column/row vectors.
	T dot(const Matk<T> &matIn) const
	{
		T sum = 0;
		T* ptr_matIn = matIn.get_ptr();

		for (int i = 0; i < ndata; i++)
			sum += (ptr_matIn[i] * ptr[i]);

		return sum;
	}

	// transpose
	// assume single channel matrix
	Matk<T> t()
	{
		Matk<T> matOut(nc, nr, 1);
		unsigned int c = 0;
		for (int j = 0; j < nc; j++)
			for (int i = 0; i < nr; i++)
				matOut(j,i) = ptr[c++];				
		return matOut;
	}

	// Generate a matrix by replicating matrix A in a block-like fashion
	// similar to matlab's repmat
	Matk<T> repmat(int ncopies_row, int ncopies_col, int ncopies_ch)
	{
		Matk<T> matOut(nr*ncopies_row, nc*ncopies_col, nch*ncopies_ch);

		int r1, r2, c1, c2, ch1, ch2;

		for (int k = 0; k < ncopies_ch; k++)
			for (int j = 0; j < ncopies_col; j++)
				for (int i = 0; i < ncopies_row; i++)
				{
					r1 = i*nr; 
					r2 = i*nr + nr - 1;
					c1 = j*nc;
					c2 = j*nc + nc - 1;
					ch1 = k*nch;
					ch2 = k*nch + nch - 1;
					matOut.ss(r1, r2, c1, c2, ch1, ch2) = *this;
				}

		return matOut;
	}

	// very fast; underlying data is the same; just change the shape
	void reshape(int nrows_new, int ncols_new, int nchannels_new)
	{
		if (nrows_new * ncols_new * nchannels_new != ndata)
		{
			printf("Reshaping not successful due to different total number of elements in the matrix.");
			return;
		}
		nr = nrows_new; nc = ncols_new; nch = nchannels_new;
	}

	// assumes that T is either: float, double or long double. Otherwise
	// will result in error
	Matk<T> round()
	{
		Matk<T> matOut(nr, nc, nch);
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < ndata; i++)
			ptr_matOut[i] = std::round( ptr[i] );

		//std::transform(ptr, ptr + ndata, ptr_matOut, [](T v) {return std::round(v); });
		//Note: I have already timed this: loop and std::transform have the about the same speed.

		return matOut;
	}

	// assumes that T is either: float, double or long double. Otherwise
	// will result in error
	void round_inplace()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i] = std::round(ptr[i]);

		//std::transform(ptr, ptr + ndata, ptr, [](T v) {return std::round(v); });
		//Note: I have already timed this: loop and std::transform have the about the same speed.
	}
	
	void zeros()
	{
		//std::fill(ptr, ptr + ndata, 0);
		std::memset(ptr, 0, sizeof(T)*ndata);
	}

	static Matk<T> zeros(int nrows_, int ncols_, int nchannels_)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		//std::fill(ptr_out, ptr_out + ndata_out, 0);
		std::memset(ptr_out, 0, sizeof(T)*ndata_out);
		return matOut;
	}

	void ones()
	{
		std::fill(ptr, ptr + ndata, 1);
		//std::memset(ptr, 1, sizeof(T)*ndata);
	}

	static Matk<T> ones(int nrows_, int ncols_, int nchannels_)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::fill(ptr_out, ptr_out + ndata_out, 1);
		//std::memset(ptr_out, 1, sizeof(T)*ndata_out);
		return matOut;
	}

	void fill(T val_fill)
	{
		std::fill(ptr, ptr + ndata, val_fill);
	}

	// Uniformly distributed pseudorandom integers between range [imin, imax]
	// similar to matlab's randi
	void randi(int imin, int imax)
	{
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(imin, imax);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed pseudorandom integers between range [imin, imax]
	// similar to matlab's randi
	static Matk<T> randi(int nrows_, int ncols_, int nchannels_, int imin, int imax)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(imin, imax);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's rand
	void rand(double dmin = 0, double dmax = 1)
	{
		std::default_random_engine eng((std::random_device())());
		std::uniform_real_distribution<double> idis(dmin, dmax);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's rand
	static Matk<T> rand(int nrows_, int ncols_, int nchannels_, double dmin = 0, double dmax = 1)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::uniform_real_distribution<double> idis(dmin, dmax);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Normally distributed random numbers 
	// similar to matlab's randn
	// for this, T should be float, double or long double, otherwise
	// undefined behaviour
	void randn(double mean = 0, double std = 1)
	{
		std::default_random_engine eng((std::random_device())());
		std::normal_distribution<double> idis(mean, std);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Normally distributed random numbers 
	// similar to matlab's randn
	static Matk<T> randn(int nrows_, int ncols_, int nchannels_, double mean = 0, double std = 1)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::normal_distribution<double> idis(mean, std);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Generate random boolean/bernoulli values, according to the discrete probability function.
	void randBernoulli(double prob_true = 0.5)
	{
		std::default_random_engine eng((std::random_device())());
		std::bernoulli_distribution idis(prob_true);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Generate random boolean/bernoulli values, according to the discrete probability function.
	static Matk<T> randBernoulli(int nrows_, int ncols_, int nchannels_, double prob_true = 0.5)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::bernoulli_distribution idis(prob_true);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Generate multinomial random values, according to the discrete probability function.
	// the input vector v represents something like votes for each category. 
	// the random number generator will normalize it and generate a distribution
	// from it. Each time a random sample is generated, it will sample generate
	// one of the values from the set {0,...,votes.size()-1} where the values corresponding
	// to higher votes are more likely to be selected.
	// if votes.size()==2, this is basically Bernoulli distribution. 
	void randDiscrete(Veck<int> votes)
	{
		std::default_random_engine eng((std::random_device())());
		std::vector<unsigned int>votes_(votes.begin(), votes.end());
		std::discrete_distribution<unsigned int> idis(votes_.begin(), votes_.end());
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Generate multinomial random values, according to the discrete probability function.
	static Matk<T> randDiscrete(int nrows_, int ncols_, int nchannels_, Veck<int> votes)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::vector<unsigned int>votes_(votes.begin(), votes.end());
		std::discrete_distribution<unsigned int> idis(votes_.begin(), votes_.end());
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	static Matk<T> fill(int nrows_, int ncols_, int nchannels_, T val_fill)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::fill(ptr_out, ptr_out + ndata_out, val_fill);
		return matOut;
	}

	// fill matrix (column wise) with decreasing/descending values 
	// (with some step) starting from a given value.
	void fill_ladder(T start_val = 0, T step = 1)
	{
		T n = start_val;
		ptr[0] = start_val; 
		std::generate(ptr+1, ptr + ndata, [&n, &step] { return n += step; });	
	}

	// fill matrix (column wise) with decreasing/descending values 
	// (with some step) starting from a given value.
	static Matk<T> fill_ladder(int nrows_, int ncols_, int nchannels_, T start_val = 0, T step = 1)
	{
		Matk<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		T n = start_val;
		ptr_out[0] = start_val;
		std::generate(ptr_out + 1, ptr_out + ndata_out, [&n, &step] { return n += step; });
		return matOut;
	}

	// fill a std vector with decreasing/descending values 
	// (with some step) starting from a given value.
	static std::vector<T> fill_ladder_stdVec(int vecSize, T start_val = 0, T step = 1)
	{
		std::vector<T> v(vecSize);
		T n = start_val;
		v[0] = start_val;
		std::generate(v.begin() + 1, v.end(), [&n, &step] { return n += step; });
		return v;
	}

	// Generate a column vector with linearly/evenly spaced values in an interval [start_val, end_val]
	// similar to matlab linspace (gives exactly the same results)
	static Matk<T> linspace(T start_val, T end_val, int nvals)
	{
		Matk<T> matOut(nvals, 1, 1);
		T* ptr_out = matOut.get_ptr();
		T step = (end_val - start_val) / (nvals - 1);
		T n = start_val;
		ptr_out[0] = start_val;
		std::generate(ptr_out + 1, ptr_out + nvals, [&n, &step] { return n += step; });
		return matOut;
	}

	// Generate a std::vector with linearly/evenly spaced values in an interval [start_val, end_val]
	// similar to matlab linspace (gives exactly the same results)
	static std::vector<T> linspace_stdVec(T start_val, T end_val, int nvals)
	{
		std::vector<T> v(nvals);
		T step = (end_val - start_val) / (nvals - 1);
		T n = start_val;
		v[0] = start_val;
		std::generate(v.begin() + 1, v.end(), [&n, &step] { return n += step; });
		return v;
	}

	// sort a matrix (2D with nchannels=1) in in the direction of row or col.
	void sort(bool sort_col, bool sort_ascend, Matk<T> &matSorted, Matk<int> &indices_sort)
	{
		matSorted.create(nr, nc, 1);
		indices_sort.create(nr, nc, 1);

		if (sort_col)
		{
			for (size_t j = 0; j < nc; j++)
			{
				Matk<int> temp_indices_sort;
				Matk<T> temp_vals_sorted;
				sort_indexes(this->gc(j), sort_ascend, temp_indices_sort, temp_vals_sorted);
				matSorted.sc(j) = temp_vals_sorted;
				indices_sort.sc(j) = temp_indices_sort;
			}
		}
		else
		{
			for (size_t i = 0; i < nr; i++)
			{
				Matk<int> temp_indices_sort;
				Matk<T> temp_vals_sorted;
				sort_indexes(this->gr(i), sort_ascend, temp_indices_sort, temp_vals_sorted);
				matSorted.sr(i) = temp_vals_sorted;
				indices_sort.sr(i) = temp_indices_sort;
			}
		}
	}

	// helper function for sort method
	// input v should be a row or column vector
	static void sort_indexes(const Matk<T> &v, bool sort_ascend, Matk<int> &indices_sort, Matk<T> &v_sorted) {

		indices_sort.create(v.nrows(), v.ncols(), 1); // either v.nrows() or v.ncols() must be 1
		v_sorted.create(v.nrows(), v.ncols(), 1); // either v.nrows() or v.ncols() must be 1

		// initialize original index locations		
		int* ptr_indices_sort = indices_sort.get_ptr();
		int lenVec = indices_sort.nelem();
		std::iota(ptr_indices_sort, ptr_indices_sort + lenVec, 0);

		// sort indexes based on comparing values in v
		if (sort_ascend)
		{
			std::sort(ptr_indices_sort, ptr_indices_sort + lenVec,
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		}
		else
		{
			std::sort(ptr_indices_sort, ptr_indices_sort + lenVec,
				[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		}

		for (size_t i = 0; i < lenVec; i++)
			v_sorted[i] = v[indices_sort[i]];
	}

	// just a experimental and not used but perfectly working well function.
	// this was written first when I wanted to write the sort method. 
	// Another method called sort_indexes that takes in Matk data types are based on
	// this. That function is directly used by the sort method, not this function.
	static std::vector<size_t> sort_indexes(const std::vector<T> &v, bool sort_ascend) {

		// initialize original index locations
		std::vector<size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		
		// sort indexes based on comparing values in v
		if (sort_ascend)
		{
			std::sort(idx.begin(), idx.end(),
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		}
		else
		{
			std::sort(idx.begin(), idx.end(),
				[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		}		
		
		return idx;
	}

	// given two equal sized matrices, return a matrix that contains the element wise max
	// of these two matrices
	static Matk<T> max(const Matk<T> &matIn1, const Matk<T> &matIn2)
	{
		Matk<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		T* ptr_in2 = matIn2.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::max(ptr_in1[i], ptr_in2[i]);
		return matOut;
	}

	// given a matrix and a scalar value, return a matrix that contains the element wise max
	// of the corresponding matrix element and the scalar
	static Matk<T> max(const Matk<T> &matIn1, T sc)
	{
		Matk<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::max(ptr_in1[i], sc);
		return matOut;
	}

	// given a a scalar value and a matrix, return a matrix that contains the element wise max
	// of the corresponding matrix element and the scalar
	static Matk<T> max(T sc, const Matk<T> &matIn1)
	{
		return max(matIn1, sc);
	}

	// given a matrix, return a matrix that contains the max along a given dimension.
	// if dim==1, returns a row vector that contains max of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains max of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains max of each channel tube.
	void max(int dim, Matk<T> &maxVals, Matk<int> &maxIndices)
	{
		switch (dim)
		{
		case 1:

			maxVals.create(1, nc, nch);
			maxIndices.create(1, nc, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t j = 0; j < nc; j++)
				{
					Matk<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					auto temp = std::max_element(ptr_cur, ptr_cur + nr);
					maxVals(0,j,k) = *temp;
					maxIndices(0, j, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 2:

			maxVals.create(nr, 1, nch);
			maxIndices.create(nr, 1, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					auto temp = std::max_element(ptr_cur, ptr_cur + nc);
					maxVals(i, 0, k) = *temp;
					maxIndices(i, 0, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 3:

			maxVals.create(nr, nc, 1);
			maxIndices.create(nr, nc, 1);
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> submat_cur = gs(i,i,j,j,0,nch);
					T* ptr_cur = submat_cur.get_ptr();
					auto temp = std::max_element(ptr_cur, ptr_cur + nch);
					maxVals(i, j) = *temp;
					maxIndices(i, j) = std::distance(ptr_cur, temp);
				}
			break;
		}
	}
	
	// given two equal sized matrices, return a matrix that contains the element wise min
	// of these two matrices
	static Matk<T> min(const Matk<T> &matIn1, const Matk<T> &matIn2)
	{
		Matk<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		T* ptr_in2 = matIn2.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::min(ptr_in1[i], ptr_in2[i]);
		return matOut;
	}

	// given a matrix and a scalar value, return a matrix that contains the element wise min
	// of the corresponding matrix element and the scalar
	static Matk<T> min(const Matk<T> &matIn1, T sc)
	{
		Matk<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::min(ptr_in1[i], sc);
		return matOut;
	}

	// given a a scalar value and a matrix, return a matrix that contains the element wise min
	// of the corresponding matrix element and the scalar
	static Matk<T> min(T sc, const Matk<T> &matIn1)
	{
		return min(matIn1, sc);
	}

	// given a matrix, return a matrix that contains the min along a given dimension.
	// if dim==1, returns a row vector that contains min of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains min of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains min of each channel tube.
	void min(int dim, Matk<T> &minVals, Matk<int> &minIndices)
	{

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		switch (dim)
		{
		case 1:

			minVals.create(1, nc, nch);
			minIndices.create(1, nc, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t j = 0; j < nc; j++)
				{
					Matk<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					auto temp = std::min_element(ptr_cur, ptr_cur + nr);
					minVals(0, j, k) = *temp;
					minIndices(0, j, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 2:

			minVals.create(nr, 1, nch);
			minIndices.create(nr, 1, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					auto temp = std::min_element(ptr_cur, ptr_cur + nc);
					minVals(i, 0, k) = *temp;
					minIndices(i, 0, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 3:

			minVals.create(nr, nc, 1);
			minIndices.create(nr, nc, 1);
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					auto temp = std::min_element(ptr_cur, ptr_cur + nch);
					minVals(i, j) = *temp;
					minIndices(i, j) = std::distance(ptr_cur, temp);
				}
			break;
		}
	}

	// given a matrix, return a matrix that contains the median along a given dimension.
	// if dim==1, returns a row vector that contains median of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains median of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains median of each channel tube.
	// Usage is the same as Matlab's median() function. Gives same results also.
	Matk<T> median(int dim)
	{

		// Implementation note:
		// uses std::nth_element which is partial sorting which is faster than sorting.
		// Finding the middle element by partial sorting. Then number of elements is odd,
		// I need to get the next element. This is done by finding the max value within
		// this partial sorted values (which are guaranteed to be less than the middle value
		// but in any order). This should still be faster than simply sorting all the elements.

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matk<T> matOut;


		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);

			int medIdx = nr / 2;
			T vn;

			if (nr % 2 != 0) // when number of elements is odd (easy case)
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t j = 0; j < nc; j++)
					{
						Matk<T> col_cur = gch(k).gc(j);
						T* ptr_cur = col_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nr);
						matOut(0, j, k) = ptr_cur[medIdx];
					}
			}

			else // when number of elements is even
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t j = 0; j < nc; j++)
					{
						Matk<T> col_cur = gch(k).gc(j);
						T* ptr_cur = col_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nr);
						vn = ptr_cur[medIdx];
						matOut(0, j, k) = 0.5*(vn + *std::max_element(ptr_cur, ptr_cur + medIdx));
					}
			}
		}

			break;

		case 2:

		{
			matOut.create(nr, 1, nch);

			int medIdx = nc / 2;
			T vn;

			if (nc % 2 != 0) // when number of elements is odd (easy case)
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t i = 0; i < nr; i++)
					{
						Matk<T> row_cur = gch(k).gr(i);
						T* ptr_cur = row_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nc);
						matOut(i, 0, k) = ptr_cur[medIdx];
					}
			}

			else // when number of elements is even
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t i = 0; i < nr; i++)
					{
						Matk<T> row_cur = gch(k).gr(i);
						T* ptr_cur = row_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nc);
						vn = ptr_cur[medIdx];
						matOut(i, 0, k) = 0.5*(vn + *std::max_element(ptr_cur, ptr_cur + medIdx));
					}
			}
		}

			break;

		case 3:

		{
			matOut.create(nr, nc, 1);

			int medIdx = nch / 2;
			T vn;

			if (nch % 2 != 0) // when number of elements is odd (easy case)
			{
				for (size_t j = 0; j < nc; j++)
					for (size_t i = 0; i < nr; i++)
					{
						Matk<T> submat_cur = gs(i, i, j, j, 0, nch);
						T* ptr_cur = submat_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nch);
						matOut(i, j) = ptr_cur[medIdx];
					}
			}

			else // when number of elements is even
			{
				for (size_t j = 0; j < nc; j++)
					for (size_t i = 0; i < nr; i++)
					{
						Matk<T> submat_cur = gs(i, i, j, j, 0, nch);
						T* ptr_cur = submat_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nch);
						vn = ptr_cur[medIdx];
						matOut(i, j) = 0.5*(vn + *std::max_element(ptr_cur, ptr_cur + medIdx));
					}
			}	
		}
			
			break;
		}

		return matOut;
	}

	// given a matrix, return a matrix that contains the mean along a given dimension.
	// if dim==1, returns a row vector that contains mean of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains mean of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains mean of each channel tube.
	// Usage is the same as Matlab's mean() function. Gives same results also.
	Matk<T> mean(int dim)
	{

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matk<T> matOut;

		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);
			
			for (size_t k = 0; k < nch; k++)
				for (size_t j = 0; j < nc; j++)
				{
					Matk<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					matOut(0, j, k) = std::accumulate(ptr_cur, ptr_cur+nr, 0.0) / (double)nr;
				}
		}

		break;

		case 2:

		{
			matOut.create(nr, 1, nch);

			for (size_t k = 0; k < nch; k++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					matOut(i, 0, k) = std::accumulate(ptr_cur, ptr_cur + nc, 0.0) / (double)nc;
				}
		}

		break;

		case 3:

		{
			matOut.create(nr, nc, 1);
	
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					matOut(i, j) = std::accumulate(ptr_cur, ptr_cur + nch, 0.0) / (double)nch;
				}
		}

		break;
		}

		return matOut;
	}

	// given a matrix, return a matrix that contains the sum along a given dimension.
	// if dim==1, returns a row vector that contains sum of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains sum of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains sum of each channel tube.
	// Usage is the same as Matlab's sum() function. Gives same results also.
	Matk<T> sum(int dim)
	{

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matk<T> matOut;

		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);

			for (size_t k = 0; k < nch; k++)
				for (size_t j = 0; j < nc; j++)
				{
					Matk<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					matOut(0, j, k) = std::accumulate(ptr_cur, ptr_cur + nr, 0.0);
				}
		}

		break;

		case 2:

		{
			matOut.create(nr, 1, nch);

			for (size_t k = 0; k < nch; k++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					matOut(i, 0, k) = std::accumulate(ptr_cur, ptr_cur + nc, 0.0);
				}
		}

		break;

		case 3:

		{
			matOut.create(nr, nc, 1);

			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					matOut(i, j) = std::accumulate(ptr_cur, ptr_cur + nch, 0.0);
				}
		}

		break;
		}

		return matOut;
	}

	// given a matrix, return a matrix that contains the std/var along a given dimension.
	// if dim==1, returns a row vector that contains std/var of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains std/var of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains std/var of each channel tube.
	// Usage is similar to Matlab's std/var function. Gives same results.
	// norm_offset can be either 1 or 0. If 1, divide by N-1 (sample estimate). If 0, divide by N.
	// if var==true, computer variance instead of standard deviation
	Matk<T> std(int dim, int norm_offset = 1, bool variance = false)
	{

		if ((norm_offset != 1) && (norm_offset != 0))
		{
			printf("norm_offset can only be 1 (division by N-1) or 0 (division by N). Using 1 as the norm_offset.\n");
			norm_offset = 1;
		}

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matk<T> matOut;
		T mean_val, sq_sum;
		std::vector<T> diff;

		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);
			diff.resize(nr);

			for (size_t k = 0; k < nch; k++)
				for (size_t j = 0; j < nc; j++)
				{
					Matk<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					mean_val = std::accumulate(ptr_cur, ptr_cur + nr, 0.0) / (double)nr;
					std::transform(ptr_cur, ptr_cur + nr, diff.begin(), [mean_val](T x) { return x - mean_val; });
					sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
					matOut(0, j, k) = sq_sum / (nr - norm_offset);
				}
		}

		break;

		case 2:

		{
			matOut.create(nr, 1, nch);
			diff.resize(nc);

			for (size_t k = 0; k < nch; k++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					mean_val = std::accumulate(ptr_cur, ptr_cur + nc, 0.0) / (double)nc;
					std::transform(ptr_cur, ptr_cur + nc, diff.begin(), [mean_val](T x) { return x - mean_val; });
					sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
					matOut(i, 0, k) = sq_sum / (nc - norm_offset);
				}
		}

		break;

		case 3:

		{
			matOut.create(nr, nc, 1);
			diff.resize(nch);

			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matk<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					mean_val = std::accumulate(ptr_cur, ptr_cur + nch, 0.0) / (double)nch;
					std::transform(ptr_cur, ptr_cur + nch, diff.begin(), [mean_val](T x) { return x - mean_val; });
					sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
					matOut(i, j) = sq_sum / (nch - norm_offset);
				}
		}

		break;
		}

		if (!variance) // if std instead of variance requested, need to take square root
		{
			T* ptr_out = matOut.get_ptr();
			for (size_t i = 0; i < matOut.nelem(); i++)
				ptr_out[i] = std::sqrt(ptr_out[i]);
		}

		return matOut;
	}

	// computes histogram of the matrix which can be size and any nchannels.
	// the algorithm just treats the given matrix as a linear array (taken in column major).
	// edges is a col/row vector that contains the edges of the bins as follows:
	// {e1,e2,e3,...,eN}. The first bin is a count of values that fall in
	// within range [e1,e2), [e2,e3), ..., [eN-1,eN].
	// nbins = N - 1. Therefore, edges vector must have at least 2 elements.
	// if I want to count integers/catogories, then I can put a dummy last edge.
	// e.g. if given data matkd e = { 2, 4, 1, 2, 5, 2 }, computing the hist
	// using e.hist({ 1,2,3,4,5 }) gives [1,3,0,2] as output.
	// This function gives same results as matlab's histcounts(X,edges).
	// which bin to chose (to increment), rather than searching linearly,
	// I use std::upper_bound which has complexity of log(n) instead of n.
	Matk<int> hist(const Matk<T> &edges)
	{
		int nedges = edges.nelem();
		int nbins = nedges - 1;		
		Matk<int> matOut(nbins, 1, 1);
		matOut.zeros();

		int *ptr_out = matOut.get_ptr();
		T *ptr_edges = edges.get_ptr();
		T *ptr_edges_end = edges.get_ptr() + nedges;

		int idx_edge_last = nedges - 1;
		T curVal;

		int idx_ub, idx_bin;

		for (size_t i = 0; i < ndata; i++)
		{
			curVal = ptr[i];			
			idx_ub = std::upper_bound(ptr_edges, ptr_edges_end, curVal) - ptr_edges;

			// handle boundary case (left most side)
			if (idx_ub == 0)
			{
				// data less than e1 (the first edge), so don't count
				if (curVal < ptr_edges[0])
					continue;				
			}					

			// handle boundary case (right most side)
			if (idx_ub == nedges)
			{
				// data greater than the last edge, so don't count
				if (curVal > ptr_edges[idx_edge_last])
					continue;
				// need to decrement since due to being at exactly edge final
				--idx_ub;					
			}

			idx_bin = idx_ub - 1;			
			++ptr_out[idx_bin];				
		}			

		return matOut;
	}

	// join this matrix with the given matrix matIn horizontally
	// if the number of rows or channels are different, max of them
	// will be taken and filled with zeros.
	Matk<T> add_cols(const Matk<T> &matIn)
	{
		int nrows_new = std::max(nr, matIn.nrows());
		int ncols_new = nc + matIn.ncols();
		int nch_new = std::max(nch, matIn.nchannels());
		Matk<T>matOut(nrows_new, ncols_new, nch_new);
		matOut.ss(0, nr - 1, 0, nc - 1, 0, nch - 1) = *this;
		matOut.ss(0, matIn.nrows() - 1, nc, ncols_new - 1, 0, matIn.nchannels() - 1) = matIn;
		return matOut;
	}

	// merge a std::vector of matrices horizontally
	// if the number of rows or channels are different, max of them
	// will be taken and filled with zeros.
	static Matk<T> merge_cols(const std::vector<Matk<T>> &vmat)
	{
		int nrows_new = 0;
		int ncols_new = 0;
		int nch_new = 0;
		int nmats = vmat.size();

		for (size_t kk = 0; kk < nmats; kk++)
		{
			nrows_new = std::max(nrows_new, vmat[kk].nrows());
			ncols_new += vmat[kk].ncols();
			nch_new = std::max(nch_new, vmat[kk].nchannels());
		}
 
		Matk<T>matOut(nrows_new, ncols_new, nch_new);
		int nc_count = 0;

		for (size_t kk = 0; kk < nmats; kk++)
		{
			matOut.ss(0, vmat[kk].nrows() - 1, nc_count, nc_count + vmat[kk].ncols() - 1, 0, vmat[kk].nchannels() - 1) = vmat[kk];
			nc_count += vmat[kk].ncols();
		}

		return matOut;
	}

	// join this matrix with the given matrix matIn vertically
	// if the number of cols or channels are different, max of them
	// will be taken and filled with zeros.
	Matk<T> add_rows(const Matk<T> &matIn)
	{		
		int nrows_new = nr + matIn.nrows();
		int ncols_new = std::max(nc, matIn.ncols());
		int nch_new = std::max(nch, matIn.nchannels());
		Matk<T>matOut(nrows_new, ncols_new, nch_new);
		matOut.ss(0, nr - 1, 0, nc - 1, 0, nch - 1) = *this;
		matOut.ss(nr, nrows_new - 1, 0, matIn.ncols() - 1, 0, matIn.nchannels() - 1) = matIn;
		return matOut;
	}

	// merge a std::vector of matrices vertically
	// if the number of cols or channels are different, max of them
	// will be taken and filled with zeros.
	static Matk<T> merge_rows(const std::vector<Matk<T>> &vmat)
	{
		int nrows_new = 0;
		int ncols_new = 0;
		int nch_new = 0;
		int nmats = vmat.size();

		for (size_t kk = 0; kk < nmats; kk++)
		{			
			nrows_new += vmat[kk].nrows();
			ncols_new = std::max(ncols_new, vmat[kk].ncols());
			nch_new = std::max(nch_new, vmat[kk].nchannels());
		}

		Matk<T>matOut(nrows_new, ncols_new, nch_new);
		int nr_count = 0;

		for (size_t kk = 0; kk < nmats; kk++)
		{
			matOut.ss(nr_count, nr_count + vmat[kk].nrows() - 1, 0, vmat[kk].ncols() - 1, 0, vmat[kk].nchannels() - 1) = vmat[kk];
			nr_count += vmat[kk].nrows();
		}

		return matOut;
	}

	// add channels to the current matrix.
	// if rows and columns of the two matrices are different, max of them
	// will be taken and filled with zeros
	Matk<T> add_channels(const Matk<T> &matIn)
	{
		int nrows_new = std::max(nr, matIn.nrows());
		int ncols_new = std::max(nc, matIn.ncols());
		int nch_new = nch + matIn.nchannels();
		Matk<T>matOut(nrows_new, ncols_new, nch_new);
		matOut.ss(0, nr - 1, 0, nc - 1, 0, nch - 1) = *this;
		matOut.ss(0, matIn.nrows() - 1, 0, matIn.ncols() - 1, nch, nch_new - 1) = matIn;
		return matOut;
	}

	// merge channels of a std::vector of matrices
	// if rows and columns of the two matrices are different, max of them
	// will be taken and filled with zeros
	static Matk<T> merge_channels(const std::vector<Matk<T>> &vmat)
	{
		int nrows_new = 0;
		int ncols_new = 0;
		int nch_new = 0;
		int nmats = vmat.size();

		for (size_t kk = 0; kk < nmats; kk++)
		{
			nrows_new = std::max(nrows_new, vmat[kk].nrows());
			ncols_new = std::max(ncols_new, vmat[kk].ncols());
			nch_new += vmat[kk].nchannels();
		}

		Matk<T>matOut(nrows_new, ncols_new, nch_new);
		int nch_count = 0;

		for (size_t kk = 0; kk < nmats; kk++)
		{
			matOut.ss(0, vmat[kk].nrows() - 1, 0, vmat[kk].ncols() - 1, nch_count, nch_count + vmat[kk].nchannels() - 1) = vmat[kk];
			nch_count += vmat[kk].nchannels();
		}

		return matOut;
	}

	// remove cols from matrix
	Matk<T> del_cols(Veck<int> indices_remove)
	{		
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// row indices that I want to keep (keep all; 1,2,...,nr)
		Veck<int> row_idxs_keep = Veck<int>::fill_ladder(nr, 0, 1);

		// channel indices that I want to keep (keep all; 1,2,...,nch)
		Veck<int> ch_idxs_keep = Veck<int>::fill_ladder(nch, 0, 1);

		// col indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> col_idxs_keep;
		// create a vector that is filled with 0,1,2,...,nc
		Veck<int> col_idxs_all = Veck<int>::fill_ladder(nc, 0, 1);
		std::set_difference(col_idxs_all.begin(), col_idxs_all.end(), 
			indices_remove.begin(), indices_remove.end(),
			std::inserter(col_idxs_keep, col_idxs_keep.begin()));

		Matk<T>matOut = gs(row_idxs_keep, Veck<int>(col_idxs_keep), ch_idxs_keep);
		
		return matOut;
	}

	// remove rows from matrix
	Matk<T> del_rows(Veck<int> indices_remove)
	{
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// col indices that I want to keep (keep all; 1,2,...,nc)
		Veck<int> col_idxs_keep = Veck<int>::fill_ladder(nc, 0, 1);

		// channel indices that I want to keep (keep all; 1,2,...,nch)
		Veck<int> ch_idxs_keep = Veck<int>::fill_ladder(nch, 0, 1);

		// row indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> row_idxs_keep;
		// create a vector that is filled with 0,1,2,...,nr
		Veck<int> row_idxs_all = Veck<int>::fill_ladder(nr, 0, 1);
		std::set_difference(row_idxs_all.begin(), row_idxs_all.end(),
			indices_remove.begin(), indices_remove.end(),
			std::inserter(row_idxs_keep, row_idxs_keep.begin()));

		Matk<T>matOut = gs(Veck<int>(row_idxs_keep), col_idxs_keep, ch_idxs_keep);

		return matOut;
	}

	// remove channels from matrix
	Matk<T> del_channels(Veck<int> indices_remove)
	{
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// row indices that I want to keep (keep all; 1,2,...,nr)
		Veck<int> row_idxs_keep = Veck<int>::fill_ladder(nr, 0, 1);

		// col indices that I want to keep (keep all; 1,2,...,nc)
		Veck<int> col_idxs_keep = Veck<int>::fill_ladder(nc, 0, 1);

		// channel indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> ch_idxs_keep;
		// create a vector that is filled with 0,1,2,...,nch
		Veck<int> ch_idxs_all = Veck<int>::fill_ladder(nch, 0, 1);
		std::set_difference(ch_idxs_all.begin(), ch_idxs_all.end(),
			indices_remove.begin(), indices_remove.end(),
			std::inserter(ch_idxs_keep, ch_idxs_keep.begin()));

		Matk<T>matOut = gs(row_idxs_keep, col_idxs_keep, Veck<int>(ch_idxs_keep));

		return matOut;
	}
	
				
	int nrows() const { return nr; }
	int ncols() const { return nc; }
	int nchannels() const { return nch; }
	int nelem() const { return ndata; }
	T* get_ptr() const { return ptr; }
	T* begin() const { return ptr; } // to ease use in STL containers/algs
	T* end() const { return ptr + ndata; } // to ease use in STL containers/algs
		
protected:
	std::unique_ptr<T[]> data;
	T *ptr; // pointer to data;
	int nr;
	int nc;
	int nch;
	int ndata; // nr*nc*nch

	void init(int nrows, int ncols, int nchannels)
	{
		nr = nrows; nc = ncols; nch = nchannels;
		ndata = nr * nc * nch;
		data = std::make_unique<T[]>(ndata);
		ptr = data.get();
	}

	void init(int nrows, int ncols, int nchannels, const T* data_external, bool external_is_row_major=false)
	{
		init(nrows, ncols, nchannels);
		if(!external_is_row_major)
			std::memcpy(ptr, data_external, sizeof(T)*ndata);
		else
		{
			unsigned int counter = 0;
			for (size_t i = 0; i < nrows; i++)
				for (size_t j = 0; j < ncols; j++)
					for (size_t k = 0; k < nchannels; k++)
						(*this)(i,j,k) = data_external[counter++];
		}
	}

};


// a class to hold the submatrix reference so that I can use it 
// as LHS to modify the submatrix for which changes will be
// reflected in the original matrix (from which the submatrix came from)
template <class T>
class Matk_ViewRef
{
public:
	Matk_ViewRef() = delete;

	Matk_ViewRef(Matk<T> &matIn, int r1_, int r2_, int c1_, int c2_, int ch1_, int ch2_)
		: mm(matIn)
	{
		r1 = r1_; r2 = r2_; c1 = c1_; c2 = c2_; ch1 = ch1_; ch2 = ch2_;
		cont_range = true;
	}

	Matk_ViewRef(Matk<T> &matIn, Veck<int>row_indices_, Veck<int>col_indices_, Veck<int>ch_indices_)
		: mm(matIn)
	{
		row_indices = row_indices_; col_indices = col_indices_; ch_indices = ch_indices_;
		cont_range = false;
	}

	// assignment operator
	void operator=(const Matk<T> &matIn)
	{	
		int nr = mm.nrows(); 
		int nc = mm.ncols();
		int nch = mm.nchannels();	
		
		T* ptr_in = matIn.get_ptr();	

		if (cont_range)
		{
			int nch_s = ch2 - ch1 + 1;
			int nc_s = c2 - c1 + 1;
			int nr_s = r2 - r1 + 1;

			T* ptr_out = mm.get_ptr();
			unsigned int cc = 0;

			// special case for when all the channels & all rows are included
			// in matlab notation, this is like mat(:,c1:c2,:)
			// this is the fastest case since the underlying is stored in
			// col major order and I can simply use a single memcpy.
			if (nch_s == nch && nr_s == nr)
			{
				std::memcpy(ptr_out + c1*nch*nr, ptr_in, sizeof(T)*(nch*nr_s));
			}

			// special case for when all the channels are included but all not rows.
			// in matlab notation, this is like mat(r1:r2,c1:c2,:).
			// this is not as fast as the first case but still much faster than the
			// general case below.
			else if (nch_s == nch && nr_s != nr)
			{
				for (unsigned int j = c1; j <= c2; j++)
				{
					std::memcpy(ptr_out + j*nch*nr + r1*nch, ptr_in + (j - c1)*nch*nr_s,  sizeof(T)*(nch*nr_s));
				}
			}

			else
			{	
				for (unsigned int k = ch1; k <= ch2; k++)
					for (unsigned int j = c1; j <= c2; j++)
						for (unsigned int i = r1; i <= r2; i++)	
							mm(i, j, k) = ptr_in[cc++];
			}
		}

		else
		{
			
			// if any range is given in the form specified in the comment above,
			// transform these range values to indices (continuous numbers)
			if ((row_indices.size() == 2) && (row_indices[0] <= 0) && (row_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((row_indices[0] == 0) && (row_indices[1] == -1)) { start_idx = 0; end_idx = nr - 1; }
				else { start_idx = abs(row_indices[0]); end_idx = abs(row_indices[1]); }
				//row_indices.resize(end_idx - start_idx + 1);
				//std::iota(row_indices.begin(), row_indices.end(), start_idx);
				row_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((col_indices.size() == 2) && (col_indices[0] <= 0) && (col_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((col_indices[0] == 0) && (col_indices[1] == -1)) { start_idx = 0; end_idx = nc - 1; }
				else { start_idx = abs(col_indices[0]); end_idx = abs(col_indices[1]); }
				//col_indices.resize(end_idx - start_idx + 1);
				//std::iota(col_indices.begin(), col_indices.end(), start_idx);
				col_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((ch_indices.size() == 2) && (ch_indices[0] <= 0) && (ch_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((ch_indices[0] == 0) && (ch_indices[1] == -1)) { start_idx = 0; end_idx = nch - 1; }
				else { start_idx = abs(ch_indices[0]); end_idx = abs(ch_indices[1]); }
				//ch_indices.resize(end_idx - start_idx + 1);
				//std::iota(ch_indices.begin(), ch_indices.end(), start_idx);
				ch_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}

			int rsize = row_indices.size();
			int csize = col_indices.size();
			int chsize = ch_indices.size();
			
			unsigned int cc = 0;
			for (unsigned int k = 0; k < chsize; k++)
				for (unsigned int j = 0; j < csize; j++)
					for (unsigned int i = 0; i < rsize; i++)
						mm(row_indices[i], col_indices[j], ch_indices[k]) = ptr_in[cc++]; 
		}
	}

	// ======================
	// public data members
	// ======================
	Matk<T> &mm; // will hold reference to the Matk object
	bool cont_range;
	int r1, r2, c1, c2, ch1, ch2; // if cont_range==1, will use these values
	Veck<int>row_indices, col_indices, ch_indices; // if cont_range==0, will use these values
};


template <class T>
class Veck_ViewRef
{
public:
	Veck_ViewRef() = delete;

	Veck_ViewRef(Veck<T> &vecIn, int r1_, int r2_)
		: mm(vecIn)
	{
		r1 = r1_; r2 = r2_;
		cont_range = true;
	}

	Veck_ViewRef(Veck<T> &vecIn, Veck<int>indices_)
		: mm(vecIn)
	{
		indices = indices_; 
		cont_range = false;
	}

	void operator=(const Veck<T> &vecIn)
	{
		if (cont_range)
		{
			for (size_t i = r1; i <= r2; i++)
				mm[i] = vecIn[i - r1];
		}

		else
		{
			int rsize = indices.size();
			for (size_t i = 0; i < rsize; i++)
				mm[indices[i]] = vecIn[i];
		}
	}

	// ======================
	// public data members
	// ======================
	Veck<T> &mm; // will hold reference to the Veck object
	bool cont_range;
	int r1, r2; // if cont_range==1, will use these values
	Veck<int>indices; // if cont_range==0, will use these values
};


template <class T>
class Veck
{
private:
	int ndata;
	std::unique_ptr<T[]> data;
	T *ptr; // pointer to data;

	void init(int size_)
	{		
		ndata = size_;
		data = std::make_unique<T[]>(ndata);
		ptr = data.get();
	}

	void init(int size_, const T* data_external)
	{
		init(size_);
		std::memcpy(ptr, data_external, sizeof(T)*ndata);
	}

public:

	T* get_ptr() const { return ptr; }
	int size() const { return ndata; }
	T* begin() const { return ptr; } // to ease use in STL containers/algs
	T* end() const { return ptr + ndata; } // to ease use in STL containers/algs

	Veck() {}
	
	~Veck() {}

	Veck(int size_)
	{
		init(size_);
	}

	Veck(int size_, T* data_external, bool copy_data=true)
	{
		if(copy_data)
			init(size_, data_external);
		else
		{
			data.reset(); // free currently owned memory by the class
			ndata = size_;
			ptr = data_external;
		}
	}
	
	Veck(const std::initializer_list<T> &iList)
	{
		init(iList.size(), iList.begin());
	}

	Veck(const std::vector<T> &vecIn)
	{
		init(vecIn.size(), vecIn.data());			
	}

	Veck(const Veck<T> &vecIn)
	{
		init(vecIn.size(), vecIn.get_ptr());
	}

	void create(int size_)
	{
		init(size_);
	}

	void create(int size_, const T* data_external)
	{
		init(size_, data_external);
	}

	void create(const std::vector<T> &vecIn)
	{
		init(vecIn.size(), vecIn.data());
	}

	// note: destroy all previous data; simply create a new vector
	void resize(int size_)
	{
		init(size_);
	}

	std::vector<T> to_StdVec()
	{
		std::vector<T> v(ndata);
		std::memcpy(v.data(), ptr, sizeof(T)*ndata);
		return v;
	}

	Matk<T> to_Matk(bool column_vec=true)
	{
		Matk<T> m;
		if (column_vec)
			m.create(ndata, 1, 1);
		else
			m.create(1, ndata, 1);
		std::memcpy(m.get_ptr(), ptr, sizeof(T)*ndata);
		return m;
	}

	void operator=(const Veck<T> &vecIn)
	{
		init(vecIn.size(), vecIn.get_ptr());
	}

	void operator=(const std::vector<T> &vecIn)
	{
		init(vecIn.size(), vecIn.data()); 
	}

	// get subvector by range (read)
	Veck<T> operator()(int r1, int r2) const
	{
		return gs(r1, r2);
	}

	// individual element access (read)
	T operator[](int linear_index) const
	{
		return ptr[linear_index];
	}

	// individual element access (write)
	T& operator[](int linear_index)
	{
		return ptr[linear_index];
	}

	// get subvector by indices (read)
	Veck<T> operator()(const Veck<int> &indices) const
	{
		return gs(indices);
	}

	void operator++(int)
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]++;
	}

	void operator++()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]++;
	}

	void operator--(int)
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]--;
	}

	void operator--()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]--;
	}
	
	// element-wise multiplication two vectors
	Veck<T> operator% (const Veck<T> &vecIn)
	{
		Veck<T> vecOut(ndata);

		T* ptr_vecIn = vecIn.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn.ndata; i++)
			ptr_vecOut[i] = ptr[i] * ptr_vecIn[i];

		return vecOut;
	}

	friend Veck<T> operator+ (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecIn2 = vecIn2.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] + ptr_vecIn2[i];

		return vecOut;
	}

	friend Veck<T> operator+ (const Veck<T> &vecIn1, T sc)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] + sc;

		return vecOut;
	}

	friend Veck<T> operator+ (T sc, const Veck<T> &vecIn1)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = sc + ptr_vecIn1[i];

		return vecOut;
	}

	friend Veck<T> operator- (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecIn2 = vecIn2.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] - ptr_vecIn2[i];

		return vecOut;
	}

	friend Veck<T> operator- (const Veck<T> &vecIn1, T sc)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] - sc;

		return vecOut;
	}

	friend Veck<T> operator- (T sc, const Veck<T> &vecIn1)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = sc - ptr_vecIn1[i];

		return vecOut;
	}

	friend Veck<int> operator== (T sc, const Veck<T> &vecIn1)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (abs(double(sc) - double(vecIn1[i])) < 0.00000001)
				indices.push_back(i);
		} 
		return Veck<int>(indices);
	}

	friend Veck<int> operator== (const Veck<T> &vecIn1, T sc)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (abs(double(sc) - double(vecIn1[i])) < 0.00000001)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator== (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (abs(double(vecIn1[i]) - double(vecIn2[i])) < 0.00000001)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator!= (T sc, const Veck<T> &vecIn1)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (abs(double(sc) - double(vecIn1[i])) > 0.00000001)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator!= (const Veck<T> &vecIn1, T sc)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (abs(double(sc) - double(vecIn1[i])) > 0.00000001)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator!= (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (abs(double(vecIn1[i]) - double(vecIn2[i])) > 0.00000001)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator>= (T sc, const Veck<T> &vecIn1)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (sc >= vecIn1[i])
				indices.push_back(i);
		}		
		return Veck<int>(indices);
	}

	friend Veck<int> operator>= (const Veck<T> &vecIn1, T sc)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] >= sc)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator>= (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] >= vecIn2[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator> (T sc, const Veck<T> &vecIn1)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (sc > vecIn1[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator> (const Veck<T> &vecIn1, T sc)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] > sc)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator> (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] > vecIn2[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}
	
	friend Veck<int> operator<= (T sc, const Veck<T> &vecIn1)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (sc <= vecIn1[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator<= (const Veck<T> &vecIn1, T sc)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] <= sc)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator<= (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] <= vecIn2[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator< (T sc, const Veck<T> &vecIn1)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (sc < vecIn1[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator< (const Veck<T> &vecIn1, T sc)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] < sc)
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}

	friend Veck<int> operator< (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		std::vector<int> indices;
		indices.reserve(vecIn1.size());
		for (int i = 0; i < vecIn1.size(); i++)
		{
			if (vecIn1[i] < vecIn2[i])
				indices.push_back(i);
		}
		return Veck<int>(indices);
	}


	// element-wise divide two vectors
	friend Veck<T> operator/ (const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecIn2 = vecIn2.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] / ptr_vecIn2[i];

		return vecOut;
	}

	// divide a vector by a scalar
	friend Veck<T> operator/ (const Veck<T> &vecIn1, T sc)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] / sc;

		return vecOut;
	}

	// divide a scalar by a vector
	friend Veck<T> operator/ (T sc, const Veck<T> &vecIn1)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = sc / ptr_vecIn1[i];

		return vecOut;
	}

	// multiply a vector and a scalar value
	friend Veck<T> operator* (const Veck<T> &vecIn1, T sc)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = ptr_vecIn1[i] * sc;

		return vecOut;
	}

	// multiply a scalar value and a vector
	friend Veck<T> operator* (T sc, const Veck<T> &vecIn1)
	{
		Veck<T> vecOut(vecIn1.ndata);

		T* ptr_vecIn1 = vecIn1.get_ptr();
		T* ptr_vecOut = vecOut.get_ptr();

		for (int i = 0; i < vecIn1.ndata; i++)
			ptr_vecOut[i] = sc * ptr_vecIn1[i];

		return vecOut;
	}

	friend std::ostream& operator<<(std::ostream &os, const Veck<T> &vecIn)
	{
		for (size_t i = 0; i < vecIn.ndata; i++)
		{
			os << std::setprecision(4) << std::fixed << vecIn[i] << "\t";
			if ((i+1) % 10 == 0) os << endl;
		}
		os << endl;
			
		return os;
	}

	// get subvector from given range
	Veck<T> gs(int r1, int r2) const
	{
		if (r1 == -1) r1 = ndata - 1; // -1 is a special notation for last one
		if (r2 == -1) r2 = ndata - 1; // -1 is a special notation for last one

		Veck<T>vecOut(r2 - r1 + 1);
		T* ptr_out = vecOut.get_ptr();
		for (size_t i = r1; i <= r2; i++)
			ptr_out[i - r1] = ptr[i];
		return vecOut;
	}	

	// get subvector from indices
	Veck<T> gs(Veck<int> indices) const
	{
		int rsize = indices.size();
		Veck<T>vecOut(rsize);
		T* ptr_out = vecOut.get_ptr();
		for (size_t i = 0; i < rsize; i++)
			ptr_out[i] = ptr[indices[i]];
		return vecOut;
	}

	// set subvector from given range (to be used as LHS)
	Veck_ViewRef<T> ss(int r1, int r2)
	{
		if (r1 == -1) r1 = ndata - 1; // -1 is a special notation for last one
		if (r2 == -1) r2 = ndata - 1; // -1 is a special notation for last one
		Veck_ViewRef<T> vref(*this, r1, r2);
		return vref;
	}

	// set subvector given indices(to be used as LHS)
	Veck_ViewRef<T> ss(Veck<int> indices)
	{
		Veck_ViewRef<T> vref(*this, indices);
		return vref;
	}

	// perform dot product between two vectors.
	T dot(const Veck<T> &vecIn) const
	{
		T sum = 0;
		T* ptr_vecIn = vecIn.get_ptr();

		for (int i = 0; i < ndata; i++)
			sum += (ptr_vecIn[i] * ptr[i]);

		return sum;
	}
	
	// Generate a longer vector by replicating vector A
	Veck<T> repmat(int ncopies)
	{
		Veck<T> vecOut(ndata*ncopies);
		int r1, r2;
		for (size_t i = 0; i < ncopies; i++)
		{
			r1 = i*ndata;
			r2 = i*ndata + ndata - 1;		
			vecOut.ss(r1, r2) = *this;
		}
		return vecOut;
	}
	
	// assumes that T is either: float, double or long double. Otherwise
	// will result in error
	Veck<T> round()
	{
		Veck<T> vecOut(ndata);
		T* ptr_vecOut = vecOut.get_ptr();
		for (size_t i = 0; i < ndata; i++)
			ptr_vecOut[i] = std::round(ptr[i]);
		return vecOut;
	}

	// assumes that T is either: float, double or long double. Otherwise
	// will result in error
	void round_inplace()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i] = std::round(ptr[i]);
	}

	void zeros()
	{
		//std::fill(begin(), end(), 0);
		std::memset(ptr, 0, sizeof(T)*ndata);
	}

	static Veck<T> zeros(int size_)
	{
		Veck<T> vecOut(size_);
		T* ptr_out = vecOut.get_ptr();
		//std::fill(vecOut.begin(), vecOut.end(), 0);
		std::memset(ptr_out, 0, sizeof(T)*size_);
		return vecOut;
	}

	void ones()
	{
		std::fill(begin(), end(), 1);
	}

	static Veck<T> ones(int size_)
	{
		Veck<T> vecOut(size_);
		std::fill(vecOut.begin(), vecOut.end(), 1);
		return vecOut;
	}

	// Uniformly distributed pseudorandom integers between range [imin, imax]
	// similar to matlab's randi
	void randi_inplace(int imin, int imax)
	{
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(imin, imax);
		std::generate(begin(), end(), [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed pseudorandom integers between range [imin, imax]
	// similar to matlab's randi
	static Veck<T> randi(int size_, int imin, int imax)
	{
		Veck<T> vecOut(size_);
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(imin, imax);
		std::generate(vecOut.begin(), vecOut.end(), [&eng, &idis] { return (T)idis(eng); });
		return vecOut;
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's rand
	void rand_inplace(double dmin = 0, double dmax = 1)
	{
		std::default_random_engine eng((std::random_device())());
		std::uniform_real_distribution<double> idis(dmin, dmax);
		std::generate(begin(), end(), [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's rand
	static Veck<T> rand(int size_, double dmin = 0, double dmax = 1)
	{
		Veck<T> vecOut(size_);
		std::default_random_engine eng((std::random_device())());
		std::uniform_real_distribution<double> idis(dmin, dmax);
		std::generate(vecOut.begin(), vecOut.end(), [&eng, &idis] { return (T)idis(eng); });
		return vecOut;
	}

	// Normally distributed random numbers between range (dmin, dmax)
	// similar to matlab's randn
	// for this, T should be float, double or long double, otherwise
	// undefined behaviour
	void randn_inplace(double mean = 0, double std = 1)
	{
		std::default_random_engine eng((std::random_device())());
		std::normal_distribution<double> idis(mean, std);
		std::generate(begin(), end(), [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's randi
	static Veck<T> randn(int size_, double mean = 0, double std = 1)
	{
		Veck<T> vecOut(size_);
		std::default_random_engine eng((std::random_device())());
		std::normal_distribution<double> idis(mean, std);
		std::generate(vecOut.begin(), vecOut.end(), [&eng, &idis] { return (T)idis(eng); });
		return vecOut;
	}

	// Generate random boolean/bernoulli values, according to the discrete probability function.
	void randBernoulli_inplace(double prob_true = 0.5)
	{
		std::default_random_engine eng((std::random_device())());
		std::bernoulli_distribution idis(prob_true);
		std::generate(begin(), end(), [&eng, &idis] { return (T)idis(eng); });
	}

	// Generate random boolean/bernoulli values, according to the discrete probability function.
	static Veck<T> randBernoulli(int size_, double prob_true = 0.5)
	{
		Veck<T> vecOut(size_);
		std::default_random_engine eng((std::random_device())());
		std::bernoulli_distribution idis(prob_true);
		std::generate(vecOut.begin(), vecOut.end(), [&eng, &idis] { return (T)idis(eng); });
		return vecOut;
	}

	// Generate multinomial random values, according to the discrete probability function.
	// the input vector v represents something like votes for each category. 
	// the random number generator will normalize it and generate a distribution
	// from it. Each time a random sample is generated, it will sample generate
	// one of the values from the set {0,...,votes.size()-1} where the values corresponding
	// to higher votes are more likely to be selected.
	// if votes.size()==2, this is basically Bernoulli distribution. 
	void randDiscrete_inplace(Veck<int> votes)
	{
		std::default_random_engine eng((std::random_device())());
		std::vector<unsigned int>votes_(votes.begin(), votes.end());
		std::discrete_distribution<unsigned int> idis(votes_.begin(), votes_.end());
		std::generate(begin(), end(), [&eng, &idis] { return (T)idis(eng); });
	}

	// Generate multinomial random values, according to the discrete probability function.
	static Veck<T> randDiscrete(int size_, Veck<int> votes)
	{
		Veck<T> vecOut(size_);
		std::default_random_engine eng((std::random_device())());
		std::vector<unsigned int>votes_(votes.begin(), votes.end());
		std::discrete_distribution<unsigned int> idis(votes_.begin(), votes_.end());
		std::generate(vecOut.begin(), vecOut.end(), [&eng, &idis] { return (T)idis(eng); });
		return vecOut;
	}
	
	void shuffle_inplace()
	{
		std::random_shuffle(begin(), end());	
	}
	
	Veck<T> shuffle()
	{
		Veck<T> vecOut = (*this);
		std::random_shuffle(vecOut.begin(), vecOut.end());
		return vecOut;
	}

	static Veck<T> fill(int size_, T val_fill)
	{
		Veck<T> vecOut(size_);
		std::fill(vecOut.begin(), vecOut.end(), val_fill);
		return vecOut;
	}

	void fill_inplace(T val_fill)
	{
		std::fill(begin(), end(), val_fill);
	}

	// fill matrix (column wise) with decreasing/descending values 
	// (with some step) starting from a given value.
	void fill_ladder_inplace(T start_val = 0, T step = 1)
	{
		T n = start_val;
		ptr[0] = start_val;
		std::generate(begin()+1, end(), [&n, &step] { return n += step; });
	}

	// fill matrix (column wise) with decreasing/descending values 
	// (with some step) starting from a given value.
	static Veck<T> fill_ladder(int size_, T start_val = 0, T step = 1)
	{
		Veck<T> vecOut(size_);
		T n = start_val;
		vecOut[0] = start_val;
		std::generate(vecOut.begin()+1, vecOut.end(), [&n, &step] { return n += step; });
		return vecOut;
	}
	
	// Generate a vector with linearly/evenly spaced values in an interval [start_val, end_val]
	// similar to matlab linspace (gives exactly the same results)
	static Veck<T> linspace(T start_val, T end_val, int nvals)
	{
		Veck<T> vecOut(nvals);
		T step = (end_val - start_val) / (nvals - 1);
		T n = start_val;
		vecOut[0] = start_val;
		std::generate(vecOut.begin() + 1, vecOut.end(), [&n, &step] { return n += step; });
		return vecOut;
	}

	static void sort(const Veck<T> &v, bool sort_ascend, Veck<int> &indices_sort, Veck<T> &v_sorted) {

		int size_ = v.size();
		indices_sort.create(size_);
		v_sorted.create(size_);
		
		// initialize original index locations		
		std::iota(indices_sort.begin(), indices_sort.end(), 0);

		// sort indexes based on comparing values in v
		if (sort_ascend)
		{
			std::sort(indices_sort.begin(), indices_sort.end(),
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		}
		else
		{
			std::sort(indices_sort.begin(), indices_sort.end(),
				[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		}

		for (size_t i = 0; i < size_; i++)
			v_sorted[i] = v[indices_sort[i]];
	}

	void sort_inplace(bool sort_ascend, Veck<int> &indices_sort) {

		indices_sort.create(ndata);

		// initialize original index locations		
		std::iota(indices_sort.begin(), indices_sort.end(), 0);

		T* v = ptr;

		// sort indexes based on comparing values in v
		if (sort_ascend)
		{
			std::sort(indices_sort.begin(), indices_sort.end(),
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		}
		else
		{
			std::sort(indices_sort.begin(), indices_sort.end(),
				[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		}

		Veck<T>temp = *this;
		for (size_t i = 0; i < ndata; i++)
			ptr[i] = temp[indices_sort[i]];
	}
	
	// given two equal sized vectors, return a vector that contains the element wise max
	// of these two vectors
	static Veck<T> max(const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		Veck<T> vecOut(vecIn1.size());
		T* ptr_out = vecOut.get_ptr();
		T* ptr_in1 = vecIn1.get_ptr();
		T* ptr_in2 = vecIn2.get_ptr();
		for (size_t i = 0; i < vecIn1.size(); i++)
			ptr_out[i] = std::max(ptr_in1[i], ptr_in2[i]);
		return vecOut;
	}

	// given a vector and a scalar value, return a vector that contains the element wise max
	// of the corresponding vector element and the scalar
	static Veck<T> max(const Veck<T> &vecIn1, T sc)
	{
		Veck<T> vecOut(vecIn1.size());
		T* ptr_out = vecOut.get_ptr();
		T* ptr_in1 = vecIn1.get_ptr();
		for (size_t i = 0; i < vecIn1.size(); i++)
			ptr_out[i] = std::max(ptr_in1[i], sc);
		return vecOut;
	}

	static Veck<T> max(T sc, const Veck<T> &vecIn1)
	{
		return max(vecIn1, sc);
	}

	T max()
	{
		T max_val_cur, max_val;
		max_val = ptr[0];
		int idx_max = 0;

		for (size_t i = 0; i < ndata; i++)
		{
			max_val_cur = std::max(ptr[i], max_val);
			if (max_val_cur > max_val)
			{
				max_val = max_val_cur;
				idx_max = i;
			}
		}

		return max_val;
	}

	T max(int &idx_max)
	{
		T max_val_cur, max_val;
		max_val = ptr[0];
		idx_max = 0;

		for (size_t i = 0; i < ndata; i++)
		{
			max_val_cur = std::max(ptr[i], max_val);
			if (max_val_cur > max_val)
			{
				max_val = max_val_cur;
				idx_max = i;
			}
		}

		return max_val;
	}

	// given two equal sized vectors, return a vector that contains the element wise min
	// of these two vectors
	static Veck<T> min(const Veck<T> &vecIn1, const Veck<T> &vecIn2)
	{
		Veck<T> vecOut(vecIn1.size());
		T* ptr_out = vecOut.get_ptr();
		T* ptr_in1 = vecIn1.get_ptr();
		T* ptr_in2 = vecIn2.get_ptr();
		for (size_t i = 0; i < vecIn1.size(); i++)
			ptr_out[i] = std::min(ptr_in1[i], ptr_in2[i]);
		return vecOut;
	}

	// given a vector and a scalar value, return a vector that contains the element wise min
	// of the corresponding vector element and the scalar
	static Veck<T> min(const Veck<T> &vecIn1, T sc)
	{
		Veck<T> vecOut(vecIn1.size());
		T* ptr_out = vecOut.get_ptr();
		T* ptr_in1 = vecIn1.get_ptr();
		for (size_t i = 0; i < vecIn1.size(); i++)
			ptr_out[i] = std::min(ptr_in1[i], sc);
		return vecOut;
	}

	static Veck<T> min(T sc, const Veck<T> &vecIn1)
	{
		return min(vecIn1, sc);
	}

	T min(int &idx_min)
	{
		T min_val_cur, min_val;
		min_val = ptr[0];
		idx_min = 0;

		for (size_t i = 0; i < ndata; i++)
		{
			min_val_cur = std::min(ptr[i], min_val);
			if (min_val_cur < min_val)
			{
				min_val = min_val_cur;
				idx_min = i;
			}
		}

		return min_val;
	}
	
	T min()
	{
		T min_val_cur, min_val;
		min_val = ptr[0];
		int idx_min = 0;

		for (size_t i = 0; i < ndata; i++)
		{
			min_val_cur = std::min(ptr[i], min_val);
			if (min_val_cur < min_val)
			{
				min_val = min_val_cur;
				idx_min = i;
			}
		}

		return min_val;
	}

	T median()
	{
		Veck<T> temp = *this; // to not modify the original array as a side effect

		int medIdx = ndata / 2;
		std::nth_element(temp.begin(), temp.begin() + medIdx, temp.end());
		T vn = temp[medIdx];

		if (ndata % 2 != 0) // when number of elements is odd (easy case)
			return vn;		
		
		// if even case
		return 0.5*(vn + *std::max_element(temp.begin(), temp.begin() + medIdx));
	}

	T mean()
	{
		return std::accumulate(begin(), end(), 0.0) / (double)ndata;
	}

	T sum()
	{
		return std::accumulate(begin(), end(), 0.0);
	}
	
	T std(int norm_offset = 1, bool variance = false)
	{
		if ((norm_offset != 1) && (norm_offset != 0))
		{
			printf("norm_offset can only be 1 (division by N-1) or 0 (division by N). Using 1 as the norm_offset.\n");
			norm_offset = 1;
		}		

		std::vector<T> diff(ndata);
		T mean_val = std::accumulate(begin(), end(), 0.0) / (double)ndata;
		std::transform(begin(), end(), diff.begin(), [mean_val](T x) { return x - mean_val; });
		T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		T var_val = sq_sum / (ndata - norm_offset);
		
		if (variance)
			return var_val;
		else
			return std::sqrt(var_val);
	}

	// computes & returns a histogram of the vector with given edges.
	// edges is a vector that contains the edges of the bins as follows:
	// {e1,e2,e3,...,eN}. The first bin is a count of values that fall in
	// within range [e1,e2), [e2,e3), ..., [eN-1,eN].
	// nbins = N - 1. Therefore, edges vector must have at least 2 elements.
	// if I want to count integers/catogories, then I can put a dummy last edge.
	// e.g. if given data matkd e = { 2, 4, 1, 2, 5, 2 }, computing the hist
	// using e.hist({ 1,2,3,4,5 }) gives [1,3,0,2] as output.
	// This function gives same results as matlab's histcounts(X,edges).
	// which bin to chose (to increment), rather than searching linearly,
	// I use std::upper_bound which has complexity of log(n) instead of n.
	Veck<int> hist(const Veck<T> &edges)
	{
		int nedges = edges.size();
		int nbins = nedges - 1;
		Veck<int> vecOut(nbins);
		vecOut.zeros();

		int *ptr_out = vecOut.get_ptr();
		T *ptr_edges = edges.get_ptr();
		T *ptr_edges_end = edges.get_ptr() + nedges;

		int idx_edge_last = nedges - 1;
		T curVal;

		int idx_ub, idx_bin;

		for (size_t i = 0; i < ndata; i++)
		{
			curVal = ptr[i];
			idx_ub = std::upper_bound(ptr_edges, ptr_edges_end, curVal) - ptr_edges;

			// handle boundary case (left most side)
			if (idx_ub == 0)
			{
				// data less than e1 (the first edge), so don't count
				if (curVal < ptr_edges[0])
					continue;
			}

			// handle boundary case (right most side)
			if (idx_ub == nedges)
			{
				// data greater than the last edge, so don't count
				if (curVal > ptr_edges[idx_edge_last])
					continue;
				// need to decrement since due to being at exactly edge final
				--idx_ub;
			}

			idx_bin = idx_ub - 1;
			++ptr_out[idx_bin];
		}

		return vecOut;
	}
	
	// join the vector with another vector and return a bigger vector
	Veck<T> join(const Veck<T> &vecIn)
	{
		int ndata_new = ndata + vecIn.size();
		Veck<T>vecOut(ndata_new);
		vecOut.ss(0, ndata - 1) = *this;
		vecOut.ss(ndata, ndata_new - 1) = vecIn;
		return vecOut;
	}

	// join the vector with another vector (modify this vector)
	void join_inplace(const Veck<T> &vecIn)
	{
		int ndata_new = ndata + vecIn.size();
		Veck<T>temp = *this; // make a copy
		init(ndata_new);
		ss(0, temp.size() - 1) = temp;
		ss(temp.size(), ndata_new - 1) = vecIn;
	}

	// join a std::vector of vectors
	static Veck<T> merge(const std::vector<Veck<T>> &vecs)
	{
		int ndata_new = 0;
		int nvecs = vecs.size();

		for (size_t kk = 0; kk < nvecs; kk++)
			ndata_new += vecs[kk].size();
		
		Veck<T>vecOut(ndata_new);
		int ndata_count = 0;

		for (size_t kk = 0; kk < nvecs; kk++)
		{
			vecOut.ss(ndata_count, ndata_count + vecs[kk].size() - 1) = vecs[kk];
			ndata_count += vecs[kk].size();
		}

		return vecOut;
	}
	
	// remove elements from vector
	Veck<T> del(Veck<int> indices_remove)
	{
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> idxs_keep;
		// create a vector that is filled with 0,1,2,...,ndata
		Veck<int> idxs_all = Veck<int>::fill_ladder(ndata, 0, 1);
		std::set_difference(idxs_all.begin(), idxs_all.end(),
			indices_remove.begin(), indices_remove.end(),
			std::inserter(idxs_keep, idxs_keep.begin()));		

		return gs(Veck<int>(idxs_keep));
	}	

	// remove elements from vector
	void del_inplace(Veck<int> indices_remove)
	{
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> idxs_keep;
		// create a vector that is filled with 0,1,2,...,ndata
		Veck<int> idxs_all = Veck<int>::fill_ladder(ndata, 0, 1);
		std::set_difference(idxs_all.begin(), idxs_all.end(),
			indices_remove.begin(), indices_remove.end(),
			std::inserter(idxs_keep, idxs_keep.begin()));

		*this = gs(Veck<int>(idxs_keep));
	}


};







template <class T>
class Matkr_ViewRef
{
public:
	Matkr_ViewRef() = delete;

	Matkr_ViewRef(Matkr<T> &matIn, int r1_, int r2_, int c1_, int c2_, int ch1_, int ch2_)
		: mm(matIn)
	{
		r1 = r1_; r2 = r2_; c1 = c1_; c2 = c2_; ch1 = ch1_; ch2 = ch2_;
		cont_range = true;
	}

	Matkr_ViewRef(Matkr<T> &matIn, Veck<int>row_indices_, Veck<int>col_indices_, Veck<int>ch_indices_)
		: mm(matIn)
	{
		row_indices = row_indices_; col_indices = col_indices_; ch_indices = ch_indices_;
		cont_range = false;
	}

	// assignment operator
	void operator=(const Matkr<T> &matIn)
	{		
		int nr = mm.nrows(); 
		int nc = mm.ncols();
		int nch = mm.nchannels();	
		
		T* ptr_in = matIn.get_ptr();		

		if (cont_range)
		{			
			int nch_s = ch2 - ch1 + 1;
			int nc_s = c2 - c1 + 1;
			int nr_s = r2 - r1 + 1;

			// Matkr<T> matOut(nr_s, nc_s, nch_s);
			T* ptr_out = mm.get_ptr();
			unsigned int cc = 0;

			// special case for when all the channels & all columns are included
			// in matlab notation, this is like mat(r1:r2,:,:)
			// this is the fastest case since the underlying is stored in
			// row major order and I can simply use a single memcpy.
			if (nch_s == nch && nc_s == nc)
			{
				std::memcpy(ptr_out + r1*nch*nc, ptr_in, sizeof(T)*(nch*nc_s));
			}

			// special case for when all the channels are included but all not columns.
			// in matlab notation, this is like mat(r1:r2,c1:c2,:).
			// this is not as fast as the first case but still much faster than the
			// general case below.
			else if (nch_s == nch && nc_s != nc)
			{
				for (unsigned int i = r1; i <= r2; i++)
				{
					std::memcpy(ptr_out + i*nch*nc + c1*nch, ptr_in + (i - r1)*nch*nc_s,  sizeof(T)*(nch*nc_s));
				}
			}
			
			// general case and will be relatively the slowest.
			// in matlab notation, this is like mat(r1:r2,c1:c2,ch1:ch2).
			else
			{	
				for (unsigned int i = r1; i <= r2; i++)
					for (unsigned int j = c1; j <= c2; j++)
						for (unsigned int k = ch1; k <= ch2; k++)
							mm(i, j, k) = ptr_in[cc++];
			}

		}

		// if not continuous range, cannot afford to use memcpy and will be 
		// much slower due to the need to perform element-wise copying.
		else 
		{						
			// if any range is given in the form specified in the comment above,
			// transform these range values to indices (continuous numbers)
			if ((row_indices.size() == 2) && (row_indices[0] <= 0) && (row_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((row_indices[0] == 0) && (row_indices[1] == -1)) { start_idx = 0; end_idx = nr - 1; }
				else { start_idx = abs(row_indices[0]); end_idx = abs(row_indices[1]); }
				//row_indices.resize(end_idx - start_idx + 1);
				//std::iota(row_indices.begin(), row_indices.end(), start_idx);
				row_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((col_indices.size() == 2) && (col_indices[0] <= 0) && (col_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((col_indices[0] == 0) && (col_indices[1] == -1)) { start_idx = 0; end_idx = nc - 1; }
				else { start_idx = abs(col_indices[0]); end_idx = abs(col_indices[1]); }
				//col_indices.resize(end_idx - start_idx + 1);
				//std::iota(col_indices.begin(), col_indices.end(), start_idx);
				col_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((ch_indices.size() == 2) && (ch_indices[0] <= 0) && (ch_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((ch_indices[0] == 0) && (ch_indices[1] == -1)) { start_idx = 0; end_idx = nch - 1; }
				else { start_idx = abs(ch_indices[0]); end_idx = abs(ch_indices[1]); }
				//ch_indices.resize(end_idx - start_idx + 1);
				//std::iota(ch_indices.begin(), ch_indices.end(), start_idx);
				ch_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}

			int rsize = row_indices.size();
			int csize = col_indices.size();
			int chsize = ch_indices.size();

			unsigned int cc = 0;
			for (unsigned int i = 0; i < rsize; i++)
				for (unsigned int j = 0; j < csize; j++)
					for (unsigned int k = 0; k < chsize; k++)
						mm(row_indices[i], col_indices[j], ch_indices[k]) = ptr_in[cc++];
		}
	}

	// ======================
	// public data members
	// ======================
	Matkr<T> &mm; // will hold reference to the Matk object
	bool cont_range;
	int r1, r2, c1, c2, ch1, ch2; // if cont_range==1, will use these values
	Veck<int>row_indices, col_indices, ch_indices; // if cont_range==0, will use these values
};


// similar to Matk, but this Maktr class is row major storage.
template <class T>
class Matkr
{
public:
	
	Matkr() {}
	
	~Matkr() {}
	
	Matkr(int nrows, int ncols, int nchannels)
	{
		init(nrows, ncols, nchannels);
	}
	
	Matkr(int nrows, int ncols)
	{
		init(nrows, ncols, 1);
	}

	Matkr(int nrows, int ncols, int nchannels, const T* data_external, bool external_is_row_major=true)
	{
		// data from outside is always copied
		init(nrows, ncols, nchannels, data_external, external_is_row_major);
	}

	Matkr(int nrows, int ncols, int nchannels, std::initializer_list<T> iList)
	{
		std::vector<T> v(iList);
		// data from outside is always copied
		init(nrows, ncols, nchannels, v.data());
	}

	// creates a column vector
	Matkr(std::initializer_list<T> iList, bool column_vec=true)
	{
		if (column_vec)
			init(iList.size(), 1, 1, iList.begin());
		else
			init(1, iList.size(), 1, iList.begin());
	}

	// row/column vector from std::vector (deep copy)
	Matkr(const std::vector<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.data()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.data()); // create row vector
	}

	// row/column vector from Veck vector (deep copy)
	Matkr(const Veck<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.get_ptr()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.get_ptr()); // create row vector
	}

	// matrix from std::vector (deep copy)
	Matkr(int nrows, int ncols, int nchannels, const std::vector<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.data());
		}
		else
			init(nrows, ncols, nchannels, vecIn.data());
	}

	Matkr(int nrows, int ncols, int nchannels, const Veck<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.get_ptr());
		}
		else
			init(nrows, ncols, nchannels, vecIn.get_ptr());
	}

	// copy constructor (deep copy); e.g. Matkr<T> m3 = m1;  //same as "Matkr<T> m3(m1);"
	// where m1 is a previously allocated/used matrix
	Matkr(const Matkr<T> &matIn)
	{
		init(matIn.nrows(), matIn.ncols(), matIn.nchannels(), matIn.get_ptr());
	}

	Matkr(const Matkr_ViewRef<T> &matIn)
	{
		create(matIn);
	}

	void create(int nrows, int ncols, int nchannels)
	{
		init(nrows, ncols, nchannels);
	}
	
	void create(int nrows, int ncols)
	{
		init(nrows, ncols, 1);
	}

	void create(int nrows, int ncols, int nchannels, const T* data_external)
	{
		// data from outside is always copied
		init(nrows, ncols, nchannels, data_external);
	}

	// create matrix from std::vector (deep copy)
	void create(int nrows, int ncols, int nchannels, const std::vector<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.data());
		}
		else
			init(nrows, ncols, nchannels, vecIn.data());
	}

	// create matrix from Veck vector (deep copy)
	void create(int nrows, int ncols, int nchannels, const Veck<T> &vecIn)
	{
		if (nrows * ncols * nchannels != vecIn.size())
		{
			printf("Input vector size and nrows, ncols and nchannels do not match.\n");
			printf("Ignoring given nrows, ncols and nchannel values.\n");
			init(vecIn.size(), 1, 1, vecIn.get_ptr());
		}
		else
			init(nrows, ncols, nchannels, vecIn.get_ptr());
	}
	
	// create row/column vector from std::vector (deep copy)
	void create(const std::vector<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.data()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.data()); // create row vector
	}

	void create(const Veck<T> &vecIn, bool column_vec = true)
	{
		if (column_vec)
			init(vecIn.size(), 1, 1, vecIn.get_ptr()); // create column vector
		else
			init(1, vecIn.size(), 1, vecIn.get_ptr()); // create row vector
	}

	void create(const Matkr_ViewRef<T> &matIn)
	{
		int nr_in = matIn.mm.nrows();
		int nc_in = matIn.mm.ncols();
		int nch_in = matIn.mm.nchannels();

		Matkr<T> &mm = matIn.mm;
		T* ptr_in = matIn.mm.get_ptr();

		if (matIn.cont_range)
		{
			int r1 = matIn.r1;
			int r2 = matIn.r2;
			int c1 = matIn.c1;
			int c2 = matIn.c2;
			int ch1 = matIn.ch1;
			int ch2 = matIn.ch2;

			int nch_s = ch2 - ch1 + 1;
			int nc_s = c2 - c1 + 1;
			int nr_s = r2 - r1 + 1;

			init(nr_s, nc_s, nch_s);

			// special case for when all the channels & all columns are included
			// in matlab notation, this is like mat(r1:r2,:,:)
			// this is the fastest case since the underlying is stored in
			// row major order and I can simply use a single memcpy.
			if (nch_s == nch_in && nc_s == nc_in)
			{
				std::memcpy(ptr, ptr_in + r1*nch_in*nc_in, sizeof(T)*(nch_in*nc_s));
			}

			// special case for when all the channels are included but all not columns.
			// in matlab notation, this is like mat(r1:r2,c1:c2,:).
			// this is not as fast as the first case but still much faster than the
			// general case below.
			else if (nch_s == nch_in && nc_s != nc_in)
			{
				for (unsigned int i = r1; i <= r2; i++)
				{
					std::memcpy(ptr + (i - r1)*nch_in*nc_s,
						ptr_in + i*nch_in*nc_in + c1*nch_in,
						sizeof(T)*(nch_in*nc_s));
				}
			}

			// general case and will be relatively the slowest.
			// in matlab notation, this is like mat(r1:r2,c1:c2,ch1:ch2).
			else
			{
				unsigned int cc = 0;
				for (unsigned int i = r1; i <= r2; i++)
					for (unsigned int j = c1; j <= c2; j++)
						for (unsigned int k = ch1; k <= ch2; k++)
							ptr[cc++] = mm(i, j, k);
			}

		}

		else
		{
			Veck<int>row_indices = matIn.row_indices;
			Veck<int>col_indices = matIn.col_indices;
			Veck<int>ch_indices = matIn.ch_indices;

			// if any range is given in the form specified in the comment above,
			// transform these range values to indices (continuous numbers)
			if ((row_indices.size() == 2) && (row_indices[0] <= 0) && (row_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((row_indices[0] == 0) && (row_indices[1] == -1)) { start_idx = 0; end_idx = nr_in - 1; }
				else { start_idx = abs(row_indices[0]); end_idx = abs(row_indices[1]); }
				//row_indices.resize(end_idx - start_idx + 1);
				//std::iota(row_indices.begin(), row_indices.end(), start_idx);
				row_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((col_indices.size() == 2) && (col_indices[0] <= 0) && (col_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((col_indices[0] == 0) && (col_indices[1] == -1)) { start_idx = 0; end_idx = nc_in - 1; }
				else { start_idx = abs(col_indices[0]); end_idx = abs(col_indices[1]); }
				//col_indices.resize(end_idx - start_idx + 1);
				//std::iota(col_indices.begin(), col_indices.end(), start_idx);
				col_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}
			if ((ch_indices.size() == 2) && (ch_indices[0] <= 0) && (ch_indices[1] <= 0))
			{
				int start_idx, end_idx;
				if ((ch_indices[0] == 0) && (ch_indices[1] == -1)) { start_idx = 0; end_idx = nch_in - 1; }
				else { start_idx = abs(ch_indices[0]); end_idx = abs(ch_indices[1]); }
				//ch_indices.resize(end_idx - start_idx + 1);
				//std::iota(ch_indices.begin(), ch_indices.end(), start_idx);
				ch_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
			}

			int rsize = row_indices.size();
			int csize = col_indices.size();
			int chsize = ch_indices.size();

			init(rsize, csize, chsize);

			unsigned int cc = 0;
			for (unsigned int i = 0; i < rsize; i++)
				for (unsigned int j = 0; j < csize; j++)
					for (unsigned int k = 0; k < chsize; k++)
						ptr[cc++] = mm(row_indices[i], col_indices[j], ch_indices[k]);
		}
	}
		
	// wrap an external contiguous array stored in row major order.
	// this class will not manage that external memory, it's the 
	// responsibility of that external source to delete that memory.
	// When wrapping, the class will release its current dynamic
	// being managed (if any) by the unique_ptr before taking on this external
	// pointer. This method is useful if I don't want to make a copy of
	// that external memory and simply use the methods of that class on it.
	// This is great for many purposes such as when the external data is huge
	// and I want to simply do some operations on it or get slices from it
	// etc.
	void wrap(int nrows, int ncols, int nchannels, T* data_external)
	{
		data.reset(); // free currently owned memory by the class
		nr = nrows; nc = ncols; nch = nchannels;
		ndata = nr * nc * nch;
		ptr = data_external; // no managing of data_external, simply a pointer pointing towards it.
	}

	// convert this matrix to std::vector
	std::vector<T> toVec()
	{
		std::vector<T> v(ndata);
		std::memcpy(v.data(), ptr, sizeof(T)*ndata);
		return v;
	}

	// convert this matrix to Veck vector
	std::vector<T> toVeck()
	{
		Veck<T> v(ndata);
		std::memcpy(v.get_ptr(), ptr, sizeof(T)*ndata);
		return v;
	}
	
	// assignment operator (deep copy); e.g. m3 = m1; // where m1 is a previously allocated/used matrix
	void operator=(const Matkr<T> &matIn)
	{
		init(matIn.nrows(), matIn.ncols(), matIn.nchannels(), matIn.get_ptr());
	}

	void operator=(const Matkr_ViewRef<T> &matIn)
	{
		create(matIn);
	}

	// create a column vector from std::vector (deep copy)
	void operator=(const std::vector<T> vecIn)
	{
		init(vecIn.size(), 1, 1, vecIn.data()); // create column vector
	}

	// create a column vector from Veck vector (deep copy)
	void operator=(const Veck<T> vecIn)
	{
		init(vecIn.size(), 1, 1, vecIn.get_ptr()); 
	}

	// individual element access (read)
	T operator()(int i, int j, int k) const
	{
		return ptr[i*nch*nc + j*nch + k];
	}

	// get submatrix by indices (read)
	Matkr<T> operator()(Veck<int> row_indices, Veck<int> col_indices, Veck<int> ch_indices) const
	{
		return gs(row_indices, col_indices, ch_indices);
	}

	// get submatrix by indices (read)
	Matkr<T> operator()(Veck<int> row_indices, Veck<int> col_indices) const
	{
		return gs(row_indices, col_indices);
	}

	// individual element access (read) (assumes k=0, i.e. referring to 1st channel)
	T operator()(int i, int j) const
	{
		return ptr[i*nc + j];
	}

	// individual element access (write)
	T& operator()(int i, int j, int k)
	{
		return ptr[i*nch*nc + j*nch + k];
	}

	// individual element access (write)  (assumes k=0, i.e. referring to 1st channel)
	T& operator()(int i, int j)
	{
		return ptr[i*nc + j];
	}
	
	// individual element access (read)
	T operator[](int linear_index) const
	{
		return ptr[linear_index];
	}

	// individual element access (write)
	T& operator[](int linear_index)
	{
		return ptr[linear_index];
	}	

	void operator++(int)
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]++;
	}

	void operator++()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]++;
	}

	void operator--(int)
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]--;
	}

	void operator--()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i]--;
	}

	// check whether another matrix is approximately equal to this matrix
	bool operator== (const Matkr<T> &matIn)
	{
		if ((matIn.nrows() != nr) || (matIn.ncols() != nc) || (matIn.nchannels() != nch))
			return false;		

		T* ptr_matIn = matIn.get_ptr();

		// check every element of matrix
		// don't check for exact equality, approximate equality with some threshold
		for (int i = 0; i < ndata; i++)
		{
			if (  abs( double(ptr_matIn[i]) - double(ptr[i]) ) > 0.00000001  )
				return false;
		}

		return true;
	}

	// check whether another matrix is NOT approximately equal to this matrix
	bool operator!= (const Matkr<T> &matIn)
	{
		return !((*this) == matIn);
	}

	// element-wise multiplication two matrices
	Matkr<T> operator% (const Matkr<T> &matIn)
	{
		Matkr<T> matOut(nr, nc, nch);

		T* ptr_matIn = matIn.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn.ndata; i++)
			ptr_matOut[i] = ptr[i] * ptr_matIn[i];

		return matOut;
	}
	
	// add two matrices
	friend Matkr<T> operator+ (const Matkr<T> &matIn1, const Matkr<T> &matIn2) 
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matIn2 = matIn2.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] + ptr_matIn2[i];

		return matOut;
	}

	// add a matrix and a number
	friend Matkr<T> operator+ (const Matkr<T> &matIn1, T sc)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] + sc;

		return matOut;
	}

	// add a number and a matrix
	friend Matkr<T> operator+ (T sc, const Matkr<T> &matIn1)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc + ptr_matIn1[i];

		return matOut;
	}

	// minus two matrices
	friend Matkr<T> operator- (const Matkr<T> &matIn1, const Matkr<T> &matIn2)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matIn2 = matIn2.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] - ptr_matIn2[i];

		return matOut;
	}

	// minus a matrix and a number
	friend Matkr<T> operator- (const Matkr<T> &matIn1, T sc)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] - sc;

		return matOut;
	}

	// minus a number and a matrix
	friend Matkr<T> operator- (T sc, const Matkr<T> &matIn1)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc - ptr_matIn1[i];

		return matOut;
	}

	// element-wise divide two matrices
	friend Matkr<T> operator/ (const Matkr<T> &matIn1, const Matkr<T> &matIn2)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matIn2 = matIn2.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] / ptr_matIn2[i];

		return matOut;
	}

	// divide a matrix by a scalar
	friend Matkr<T> operator/ (const Matkr<T> &matIn1, T sc)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();		
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] / sc;

		return matOut;
	}

	// divide a scalar by a matrix
	friend Matkr<T> operator/ (T sc, const Matkr<T> &matIn1)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc / ptr_matIn1[i];

		return matOut;
	}

	// matrix multiplication of two matrices
	// assume nchannels == 1 for both matrices
	friend Matkr<T> operator* (const Matkr<T> &matIn1, const Matkr<T> &matIn2)
	{
		int nr_new = matIn1.nr;
		int nc_new = matIn2.nc;

		Matkr<T> matOut(nr_new, nc_new, 1);

		T* ptr_out = matOut.get_ptr();

		unsigned int c = 0;

		for (int i = 0; i < nr_new; i++)
			for (int j = 0; j < nc_new; j++)			
				ptr_out[c++] = matIn1.gr(i).dot(matIn2.gc(j));				

		return matOut;
	}

	// multiply a matrix and a scalar value
	friend Matkr<T> operator* (const Matkr<T> &matIn1, T sc)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = ptr_matIn1[i] * sc;

		return matOut;
	}

	// multiply a scalar value and a matrix
	friend Matkr<T> operator* (T sc, const Matkr<T> &matIn1)
	{
		Matkr<T> matOut(matIn1.nr, matIn1.nc, matIn1.nch);

		T* ptr_matIn1 = matIn1.get_ptr();
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < matIn1.ndata; i++)
			ptr_matOut[i] = sc * ptr_matIn1[i];

		return matOut;
	}

	friend std::ostream& operator<<(std::ostream &os, const Matkr<T> &matIn)
	{
		for (unsigned int k = 0; k < matIn.nchannels(); k++)
		{
			os << "Channel " << k + 1 << ":" << endl;
			for (unsigned int i = 0; i < matIn.nrows(); i++)
			{
				for (unsigned int j = 0; j < matIn.ncols(); j++)
				{
					os << std::setprecision(4) << std::fixed << matIn(i, j, k) << "\t";
				}
				os << endl;
			}
			os << "=========================" << endl;
		}
		return os;
	}

	// get submatrix from given range
	Matkr<T> gs(int r1, int r2, int c1, int c2, int ch1, int ch2) const
	{
		if (r1 == -1) r1 = nr - 1; // -1 is a special notation for last one
		if (r2 == -1) r2 = nr - 1; // -1 is a special notation for last one
		if (c1 == -1) c1 = nc - 1; // -1 is a special notation for last one
		if (c2 == -1) c2 = nc - 1; // -1 is a special notation for last one
		if (ch1 == -1) ch1 = nch - 1; // -1 is a special notation for last one
		if (ch2 == -1) ch2 = nch - 1; // -1 is a special notation for last one
		
		int nch_s = ch2 - ch1 + 1;
		int nc_s = c2 - c1 + 1;
		int nr_s = r2 - r1 + 1;

		Matkr<T> matOut(nr_s, nc_s, nch_s);
		T* ptr_out = matOut.get_ptr();		
		
		// special case for when all the channels & all columns are included
		// in matlab notation, this is like mat(r1:r2,:,:)
		// this is the fastest case since the underlying is stored in
		// row major order and I can simply use a single memcpy.
		if(nch_s==nch && nc_s==nc) 
		{		
			std::memcpy(ptr_out, ptr+r1*nch*nc, sizeof(T)*(nch*nc_s));
		}
		
		// special case for when all the channels are included but all not columns.
		// in matlab notation, this is like mat(r1:r2,c1:c2,:).
		// this is not as fast as the first case but still much faster than the
		// general case below.
		else if (nch_s==nch && nc_s!=nc) 
		{
			for (unsigned int i = r1; i <= r2; i++)
			{
				std::memcpy(ptr_out+(i-r1)*nch*nc_s, ptr+i*nch*nc+c1*nch, sizeof(T)*(nch*nc_s));
			}
		}
		
		// general case and will be the slowest.
		// in matlab notation, this is like mat(r1:r2,c1:c2,ch1:ch2).
		else
		{
			unsigned int cc = 0;
			for (unsigned int i = r1; i <= r2; i++)
				for (unsigned int j = c1; j <= c2; j++)
					for (unsigned int k = ch1; k <= ch2; k++)				
						ptr_out[cc++] = (*this)(i, j, k);								
		}
						
		return matOut;
	}
	
	// get submatrix given indices & range combination
	// apart from vector of index values,
	// optionally allow range specification which is denoted by two-value
	// vector, both of which are <= zero. 
	// e.g. {-3,-7} means continuous range from 3 to 7.
	// e.g. {0,10} means continous range from 0 to 10.
	// a special case is {0,-1} which denotes entire range, i.e. 0 to nrows/ncols/nch
	// does deep copy
	Matkr<T> gs(Veck<int> row_indices, Veck<int> col_indices, Veck<int> ch_indices) const
	{

		// if any range is given in the form specified in the comment above,
		// transform these range values to indices (continuous numbers)
		if ((row_indices.size()==2) && (row_indices[0] <= 0) && (row_indices[1] <= 0))
		{
			int start_idx, end_idx;
			if ((row_indices[0] == 0) && (row_indices[1] == -1)) { start_idx = 0; end_idx = nr - 1; }
			else { start_idx = abs(row_indices[0]); end_idx = abs(row_indices[1]); }				
			//row_indices.resize(end_idx - start_idx + 1);
			//std::iota(row_indices.begin(), row_indices.end(), start_idx); 
			row_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
		}
		if ((col_indices.size() == 2) && (col_indices[0] <= 0) && (col_indices[1] <= 0))
		{
			int start_idx, end_idx;
			if ((col_indices[0] == 0) && (col_indices[1] == -1)) { start_idx = 0; end_idx = nc - 1; }
			else { start_idx = abs(col_indices[0]); end_idx = abs(col_indices[1]); }				
			//col_indices.resize(end_idx - start_idx + 1);
			//std::iota(col_indices.begin(), col_indices.end(), start_idx);
			col_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
		}
		if ((ch_indices.size() == 2) && (ch_indices[0] <= 0) && (ch_indices[1] <= 0))
		{
			int start_idx, end_idx;
			if ((ch_indices[0] == 0) && (ch_indices[1] == -1)) { start_idx = 0; end_idx = nch - 1; }
			else { start_idx = abs(ch_indices[0]); end_idx = abs(ch_indices[1]); }				
			//ch_indices.resize(end_idx - start_idx + 1);
			//std::iota(ch_indices.begin(), ch_indices.end(), start_idx);
			ch_indices = Veck<int>::fill_ladder(end_idx - start_idx + 1, start_idx, 1);
		}
		
		int rsize = row_indices.size();
		int csize = col_indices.size();
		int chsize = ch_indices.size();

		Matkr<T>matOut(rsize, csize, chsize);

		T* ptr_out = matOut.get_ptr();
		unsigned int counter = 0;

		for (unsigned int i = 0; i < rsize; i++)
			for (unsigned int j = 0; j < csize; j++)
				for (unsigned int k = 0; k < chsize; k++)	
					ptr_out[counter++] = (*this)(row_indices[i], col_indices[j], ch_indices[k]);

		return matOut;
	}
	
	// get submatrix from given range (assumes all channels requested)
	Matkr<T> gs(int r1, int r2, int c1, int c2) const
	{
		return gs(r1, r2, c1, c2, 0, -1);
	}
	
	// get submatrix from given indices (assumes all channels requested)
	Matkr<T> gs(Veck<int> row_indices, Veck<int> col_indices) const
	{
		Veck<int>ch_indices = {0, -1};
		return gs(row_indices, col_indices, ch_indices);
	}

	// get specific columns of matrix from range c1:c2
	// assumes all rows and channels are requested
	Matkr<T> gc(int c1, int c2) const
	{
		return gs(0, -1, c1, c2, 0, -1);
	}

	// get specific columns of matrix from given indices
	// assumes all rows and channels are requested
	Matkr<T> gc(Veck<int> col_indices) const
	{
		return gs({ 0, -1 }, col_indices, { 0, -1 });
	}

	// get a specific column of matrix
	// assumes all rows and channels are requested
	Matkr<T> gc(int c) const
	{
		return gs(0, -1, c, c, 0, -1);
	}

	// get specific rows of matrix from range r1:r2
	// assumes all cols and channels are requested
	Matkr<T> gr(int r1, int r2)  const
	{
		return gs(r1, r2, 0, -1, 0, -1);
	}

	// get specific rows of matrix from given indices
	// assumes all cols and channels are requested
	Matkr<T> gr(Veck<int> row_indices) const
	{
		return gs(row_indices, { 0, -1 }, { 0, -1 });
	}

	// get a specific row of matrix
	// assumes all cols and channels are requested
	Matkr<T> gr(int r) const
	{
		return gs(r, r, 0, -1, 0, -1);
	}

	// get specific channels of matrix from range ch1:ch2
	// assumes all rows and cols are requested
	Matkr<T> gch(int ch1, int ch2) const
	{
		return gs(0, -1, 0, -1, ch1, ch2);
	}

	// get specific channels of matrix from given indices
	// assumes all rows and cols are requested
	Matkr<T> gch(Veck<int> ch_indices) const
	{
		return gs({ 0, -1 }, { 0, -1 }, ch_indices);
	}

	// get a specific channel of matrix
	// assumes all rows and cols are requested
	Matkr<T> gch(int ch) const
	{
		return gs(0, -1, 0, -1, ch, ch);
	}

	// get ref to submatrix from given range
	// can also be used as LHS to set values in the matrix
	Matkr_ViewRef<T> ss(int r1, int r2, int c1, int c2, int ch1, int ch2)
	{
		if (r1 == -1) r1 = nr - 1; // -1 is a special notation for last one
		if (r2 == -1) r2 = nr - 1; // -1 is a special notation for last one
		if (c1 == -1) c1 = nc - 1; // -1 is a special notation for last one
		if (c2 == -1) c2 = nc - 1; // -1 is a special notation for last one
		if (ch1 == -1) ch1 = nch - 1; // -1 is a special notation for last one
		if (ch2 == -1) ch2 = nch - 1; // -1 is a special notation for last one

		return Matkr_ViewRef<T>(*this, r1, r2, c1, c2, ch1, ch2);		
	}

	// get ref to submatrix from given range 
	// can also be used as LHS to set values in the matrix
	// assumes all channels are requested
	Matkr_ViewRef<T> ss(int r1, int r2, int c1, int c2)
	{
		return ss(r1, r2, c1, c2, 0, -1);
	}

	// get ref to submatrix given indices & range combination.
	// can also be used as LHS to set values in the matrix
	Matkr_ViewRef<T> ss(Veck<int> row_indices, Veck<int> col_indices, Veck<int> ch_indices)
	{
		return Matkr_ViewRef<T>(*this, row_indices, col_indices, ch_indices);
	}

	// get ref to submatrix given indices & range combination.
	// can also be used as LHS to set values in the matrix
	// assumes all channels are requested.
	Matkr_ViewRef<T> ss(Veck<int> row_indices, Veck<int> col_indices)
	{
		return ss(row_indices, col_indices, { 0, -1 });
	}
	
	// get ref to specific columns of matrix from range c1:c2 
	// can also be used as LHS to set values in the matrix
	// assumes all rows and channels are requested.
	Matkr_ViewRef<T> sc(int c1, int c2)
	{
		return ss(0, -1, c1, c2, 0, -1);
	}

	// get ref to specific columns of matrix from col indices
	// can also be used as LHS to set values in the matrix
	// assumes all rows and channels are requested.
	Matkr_ViewRef<T> sc(Veck<int> col_indices)
	{
		return ss({ 0, -1 }, col_indices, { 0, -1 });
	}

	// get ref to a specific column of matrix from a col index
	// can also be used as LHS to set values in the matrix
	// assumes all rows and channels are requested.
	Matkr_ViewRef<T> sc(int c)
	{
		return ss(0, -1, c, c, 0, -1);
	}

	// get ref to a specific rows of matrix from range r1:r2 
	// can also be used as LHS to set values in the matrix
	// assumes all cols and channels are requested.
	Matkr_ViewRef<T> sr(int r1, int r2)
	{
		return ss(r1, r2, 0, -1, 0, -1);
	}

	// get ref to a specific rows of matrix from row indices
	// can also be used as LHS to set values in the matrix
	// assumes all cols and channels are requested.	
	Matkr_ViewRef<T> sr(Veck<int> row_indices)
	{
		return ss(row_indices, { 0, -1 }, { 0, -1 });
	}
	
	// get ref to a specific row of matrix
	// can also be used as LHS to set values in the matrix
	// assumes all cols and channels are requested.	
	Matkr_ViewRef<T> sr(int r)
	{
		return ss(r, r, 0, -1, 0, -1);
	}

	// get ref to a specific channels of matrix from range ch1:ch2
	// can also be used as LHS to set values in the matrix
	// assumes all rows and cols are requested.		
	Matkr_ViewRef<T> sch(int ch1, int ch2)
	{
		return ss(0, -1, 0, -1, ch1, ch2);
	}

	// get ref to a specific channels of matrix from channel indices
	// can also be used as LHS to set values in the matrix
	// assumes all rows and cols are requested.	
	Matkr_ViewRef<T> sch(Veck<int> ch_indices)
	{
		return ss({ 0, -1 }, { 0, -1 }, ch_indices);
	}

	// get ref to a specific channel of matrix
	// can also be used as LHS to set values in the matrix
	// assumes all rows and cols are requested.	
	Matkr_ViewRef<T> sch(int ch)
	{
		return ss(0, -1, 0, -1, ch, ch);
	}

	// get a column/row vector out of the matrix by flattening
	Matkr<T> vectorize(bool col_vec = true) const
	{
		if (col_vec)
			return Matkr<T>(ndata, 1, 1, ptr);
		else
			return Matkr<T>(1, ndata, 1, ptr);
	}

	// perform dot product between two column/row vectors.
	T dot(const Matkr<T> &matIn) const
	{
		T sum = 0;
		T* ptr_matIn = matIn.get_ptr();

		for (int i = 0; i < ndata; i++)
			sum += (ptr_matIn[i] * ptr[i]);

		return sum;
	}

	// transpose
	// assume single channel matrix
	Matkr<T> t()
	{
		Matkr<T> matOut(nc, nr, 1);
		unsigned int c = 0;
		for (int i = 0; i < nr; i++)
			for (int j = 0; j < nc; j++)			
				matOut(j,i) = ptr[c++];				
		return matOut;
	}

	// Generate a matrix by replicating matrix A in a block-like fashion
	// similar to matlab's repmat
	Matkr<T> repmat(int ncopies_row, int ncopies_col, int ncopies_ch)
	{
		Matkr<T> matOut(nr*ncopies_row, nc*ncopies_col, nch*ncopies_ch);

		int r1, r2, c1, c2, ch1, ch2;

		for (int i = 0; i < ncopies_row; i++)
			for (int j = 0; j < ncopies_col; j++)
				for (int k = 0; k < ncopies_ch; k++)				
				{
					r1 = i*nr; 
					r2 = i*nr + nr - 1;
					c1 = j*nc;
					c2 = j*nc + nc - 1;
					ch1 = k*nch;
					ch2 = k*nch + nch - 1;
					matOut.ss(r1, r2, c1, c2, ch1, ch2) = *this;
				}

		return matOut;
	}

	// very fast; underlying data is the same; just change the shape
	void reshape(int nrows_new, int ncols_new, int nchannels_new)
	{
		if (nrows_new * ncols_new * nchannels_new != ndata)
		{
			printf("Reshaping not successful due to different total number of elements in the matrix.");
			return;
		}
		nr = nrows_new; nc = ncols_new; nch = nchannels_new;
	}

	// assumes that T is either: float, double or long double. Otherwise
	// will result in error
	Matkr<T> round()
	{
		Matkr<T> matOut(nr, nc, nch);
		T* ptr_matOut = matOut.get_ptr();

		for (int i = 0; i < ndata; i++)
			ptr_matOut[i] = std::round( ptr[i] );

		//std::transform(ptr, ptr + ndata, ptr_matOut, [](T v) {return std::round(v); });
		//Note: I have already timed this: loop and std::transform have the about the same speed.

		return matOut;
	}

	// assumes that T is either: float, double or long double. Otherwise
	// will result in error
	void round_inplace()
	{
		for (int i = 0; i < ndata; i++)
			ptr[i] = std::round(ptr[i]);

		//std::transform(ptr, ptr + ndata, ptr, [](T v) {return std::round(v); });
		//Note: I have already timed this: loop and std::transform have the about the same speed.
	}
	
	void zeros()
	{
		//std::fill(ptr, ptr + ndata, 0);
		std::memset(ptr, 0, sizeof(T)*ndata);
	}

	static Matkr<T> zeros(int nrows_, int ncols_, int nchannels_)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		//std::fill(ptr_out, ptr_out + ndata_out, 0);
		std::memset(ptr_out, 0, sizeof(T)*ndata_out);
		return matOut;
	}

	void ones()
	{
		std::fill(ptr, ptr + ndata, 1);
		//std::memset(ptr, 1, sizeof(T)*ndata);
	}

	static Matkr<T> ones(int nrows_, int ncols_, int nchannels_)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::fill(ptr_out, ptr_out + ndata_out, 1);
		//std::memset(ptr_out, 1, sizeof(T)*ndata_out);
		return matOut;
	}

	void fill(T val_fill)
	{
		std::fill(ptr, ptr + ndata, val_fill);
	}

	// Uniformly distributed pseudorandom integers between range [imin, imax]
	// similar to matlab's randi
	void randi(int imin, int imax)
	{
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(imin, imax);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed pseudorandom integers between range [imin, imax]
	// similar to matlab's randi
	static Matkr<T> randi(int nrows_, int ncols_, int nchannels_, int imin, int imax)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(imin, imax);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's rand
	void rand(double dmin = 0, double dmax = 1)
	{
		std::default_random_engine eng((std::random_device())());
		std::uniform_real_distribution<double> idis(dmin, dmax);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's rand
	static Matkr<T> rand(int nrows_, int ncols_, int nchannels_, double dmin = 0, double dmax = 1)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::uniform_real_distribution<double> idis(dmin, dmax);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Normally distributed random numbers between range (dmin, dmax)
	// similar to matlab's randn
	// for this, T should be float, double or long double, otherwise
	// undefined behaviour
	void randn(double mean = 0, double std = 1)
	{
		std::default_random_engine eng((std::random_device())());
		std::normal_distribution<double> idis(mean, std);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Uniformly distributed random numbers between range (dmin, dmax)
	// similar to matlab's randi
	static Matkr<T> randn(int nrows_, int ncols_, int nchannels_, double mean = 0, double std = 1)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::normal_distribution<double> idis(mean, std);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Generate random boolean/bernoulli values, according to the discrete probability function.
	void randBernoulli(double prob_true = 0.5)
	{
		std::default_random_engine eng((std::random_device())());
		std::bernoulli_distribution idis(prob_true);
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Generate random boolean/bernoulli values, according to the discrete probability function.
	static Matkr<T> randBernoulli(int nrows_, int ncols_, int nchannels_, double prob_true = 0.5)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::bernoulli_distribution idis(prob_true);
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	// Generate multinomial random values, according to the discrete probability function.
	// the input vector v represents something like votes for each category. 
	// the random number generator will normalize it and generate a distribution
	// from it. Each time a random sample is generated, it will sample generate
	// one of the values from the set {0,...,votes.size()-1} where the values corresponding
	// to higher votes are more likely to be selected.
	// if votes.size()==2, this is basically Bernoulli distribution. 
	void randDiscrete(Veck<int> votes)
	{
		std::default_random_engine eng((std::random_device())());
		std::vector<unsigned int>votes_(votes.begin(), votes.end());
		std::discrete_distribution<unsigned int> idis(votes_.begin(), votes_.end());
		std::generate(ptr, ptr + ndata, [&eng, &idis] { return (T)idis(eng); });
	}

	// Generate multinomial random values, according to the discrete probability function.
	static Matkr<T> randDiscrete(int nrows_, int ncols_, int nchannels_, Veck<int> votes)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::default_random_engine eng((std::random_device())());
		std::vector<unsigned int>votes_(votes.begin(), votes.end());
		std::discrete_distribution<unsigned int> idis(votes_.begin(), votes_.end());
		std::generate(ptr_out, ptr_out + ndata_out, [&eng, &idis] { return (T)idis(eng); });
		return matOut;
	}

	static Matkr<T> fill(int nrows_, int ncols_, int nchannels_, T val_fill)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		std::fill(ptr_out, ptr_out + ndata_out, val_fill);
		return matOut;
	}

	// fill matrix (column wise) with decreasing/descending values 
	// (with some step) starting from a given value.
	void fill_ladder(T start_val = 0, T step = 1)
	{
		T n = start_val;
		ptr[0] = start_val; 
		std::generate(ptr+1, ptr + ndata, [&n, &step] { return n += step; });	
	}

	// fill matrix (column wise) with decreasing/descending values 
	// (with some step) starting from a given value.
	static Matkr<T> fill_ladder(int nrows_, int ncols_, int nchannels_, T start_val = 0, T step = 1)
	{
		Matkr<T> matOut(nrows_, ncols_, nchannels_);
		T* ptr_out = matOut.get_ptr();
		unsigned int ndata_out = nrows_ * ncols_ * nchannels_;
		T n = start_val;
		ptr_out[0] = start_val;
		std::generate(ptr_out + 1, ptr_out + ndata_out, [&n, &step] { return n += step; });
		return matOut;
	}

	// fill a std vector with decreasing/descending values 
	// (with some step) starting from a given value.
	static std::vector<T> fill_ladder_stdVec(int vecSize, T start_val = 0, T step = 1)
	{
		std::vector<T> v(vecSize);
		T n = start_val;
		v[0] = start_val;
		std::generate(v.begin() + 1, v.end(), [&n, &step] { return n += step; });
		return v;
	}

	// Generate a column vector with linearly/evenly spaced values in an interval [start_val, end_val]
	// similar to matlab linspace (gives exactly the same results)
	static Matkr<T> linspace(T start_val, T end_val, int nvals)
	{
		Matkr<T> matOut(nvals, 1, 1);
		T* ptr_out = matOut.get_ptr();
		T step = (end_val - start_val) / (nvals - 1);
		T n = start_val;
		ptr_out[0] = start_val;
		std::generate(ptr_out + 1, ptr_out + nvals, [&n, &step] { return n += step; });
		return matOut;
	}

	// Generate a std::vector with linearly/evenly spaced values in an interval [start_val, end_val]
	// similar to matlab linspace (gives exactly the same results)
	static std::vector<T> linspace_stdVec(T start_val, T end_val, int nvals)
	{
		std::vector<T> v(nvals);
		T step = (end_val - start_val) / (nvals - 1);
		T n = start_val;
		v[0] = start_val;
		std::generate(v.begin() + 1, v.end(), [&n, &step] { return n += step; });
		return v;
	}

	// sort a matrix (2D with nchannels=1) in in the direction of row or col.
	void sort(bool sort_col, bool sort_ascend, Matkr<T> &matSorted, Matkr<int> &indices_sort)
	{
		matSorted.create(nr, nc, 1);
		indices_sort.create(nr, nc, 1);

		if (sort_col)
		{
			for (size_t j = 0; j < nc; j++)
			{
				Matkr<int> temp_indices_sort;
				Matkr<T> temp_vals_sorted;
				sort_indexes(this->gc(j), sort_ascend, temp_indices_sort, temp_vals_sorted);
				matSorted.sc(j) = temp_vals_sorted;
				indices_sort.sc(j) = temp_indices_sort;
			}
		}
		else
		{
			for (size_t i = 0; i < nr; i++)
			{
				Matkr<int> temp_indices_sort;
				Matkr<T> temp_vals_sorted;
				sort_indexes(this->gr(i), sort_ascend, temp_indices_sort, temp_vals_sorted);
				matSorted.sr(i) = temp_vals_sorted;
				indices_sort.sr(i) = temp_indices_sort;
			}
		}
	}

	// helper function for sort method
	// input v should be a row or column vector
	static void sort_indexes(const Matkr<T> &v, bool sort_ascend, Matkr<int> &indices_sort, Matkr<T> &v_sorted) {

		indices_sort.create(v.nrows(), v.ncols(), 1); // either v.nrows() or v.ncols() must be 1
		v_sorted.create(v.nrows(), v.ncols(), 1); // either v.nrows() or v.ncols() must be 1

		// initialize original index locations		
		int* ptr_indices_sort = indices_sort.get_ptr();
		int lenVec = indices_sort.nelem();
		std::iota(ptr_indices_sort, ptr_indices_sort + lenVec, 0);

		// sort indexes based on comparing values in v
		if (sort_ascend)
		{
			std::sort(ptr_indices_sort, ptr_indices_sort + lenVec,
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		}
		else
		{
			std::sort(ptr_indices_sort, ptr_indices_sort + lenVec,
				[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		}

		for (size_t i = 0; i < lenVec; i++)
			v_sorted[i] = v[indices_sort[i]];
	}

	// just a experimental and not used but perfectly working well function.
	// this was written first when I wanted to write the sort method. 
	// Another method called sort_indexes that takes in Matkr data types are based on
	// this. That function is directly used by the sort method, not this function.
	static std::vector<size_t> sort_indexes(const std::vector<T> &v, bool sort_ascend) {

		// initialize original index locations
		std::vector<size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		
		// sort indexes based on comparing values in v
		if (sort_ascend)
		{
			std::sort(idx.begin(), idx.end(),
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		}
		else
		{
			std::sort(idx.begin(), idx.end(),
				[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		}		
		
		return idx;
	}

	// given two equal sized matrices, return a matrix that contains the element wise max
	// of these two matrices
	static Matkr<T> max(const Matkr<T> &matIn1, const Matkr<T> &matIn2)
	{
		Matkr<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		T* ptr_in2 = matIn2.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::max(ptr_in1[i], ptr_in2[i]);
		return matOut;
	}

	// given a matrix and a scalar value, return a matrix that contains the element wise max
	// of the corresponding matrix element and the scalar
	static Matkr<T> max(const Matkr<T> &matIn1, T sc)
	{
		Matkr<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::max(ptr_in1[i], sc);
		return matOut;
	}

	// given a a scalar value and a matrix, return a matrix that contains the element wise max
	// of the corresponding matrix element and the scalar
	static Matkr<T> max(T sc, const Matkr<T> &matIn1)
	{
		return max(matIn1, sc);
	}

	// given a matrix, return a matrix that contains the max along a given dimension.
	// if dim==1, returns a row vector that contains max of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains max of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains max of each channel tube.
	void max(int dim, Matkr<T> &maxVals, Matkr<int> &maxIndices)
	{
		switch (dim)
		{
		case 1:

			maxVals.create(1, nc, nch);
			maxIndices.create(1, nc, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t j = 0; j < nc; j++)
				{
					Matkr<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					auto temp = std::max_element(ptr_cur, ptr_cur + nr);
					maxVals(0,j,k) = *temp;
					maxIndices(0, j, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 2:

			maxVals.create(nr, 1, nch);
			maxIndices.create(nr, 1, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					auto temp = std::max_element(ptr_cur, ptr_cur + nc);
					maxVals(i, 0, k) = *temp;
					maxIndices(i, 0, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 3:

			maxVals.create(nr, nc, 1);
			maxIndices.create(nr, nc, 1);
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> submat_cur = gs(i,i,j,j,0,nch);
					T* ptr_cur = submat_cur.get_ptr();
					auto temp = std::max_element(ptr_cur, ptr_cur + nch);
					maxVals(i, j) = *temp;
					maxIndices(i, j) = std::distance(ptr_cur, temp);
				}
			break;
		}
	}
	
	// given two equal sized matrices, return a matrix that contains the element wise min
	// of these two matrices
	static Matkr<T> min(const Matkr<T> &matIn1, const Matkr<T> &matIn2)
	{
		Matkr<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		T* ptr_in2 = matIn2.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::min(ptr_in1[i], ptr_in2[i]);
		return matOut;
	}

	// given a matrix and a scalar value, return a matrix that contains the element wise min
	// of the corresponding matrix element and the scalar
	static Matkr<T> min(const Matkr<T> &matIn1, T sc)
	{
		Matkr<T> matOut(matIn1.nrows(), matIn1.ncols(), matIn1.nchannels());
		T* ptr_out = matOut.get_ptr();
		T* ptr_in1 = matIn1.get_ptr();
		for (size_t i = 0; i < matIn1.nelem(); i++)
			ptr_out[i] = std::min(ptr_in1[i], sc);
		return matOut;
	}

	// given a a scalar value and a matrix, return a matrix that contains the element wise min
	// of the corresponding matrix element and the scalar
	static Matkr<T> min(T sc, const Matkr<T> &matIn1)
	{
		return min(matIn1, sc);
	}

	// given a matrix, return a matrix that contains the min along a given dimension.
	// if dim==1, returns a row vector that contains min of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains min of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains min of each channel tube.
	void min(int dim, Matkr<T> &minVals, Matkr<int> &minIndices)
	{

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		switch (dim)
		{
		case 1:

			minVals.create(1, nc, nch);
			minIndices.create(1, nc, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t j = 0; j < nc; j++)
				{
					Matkr<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					auto temp = std::min_element(ptr_cur, ptr_cur + nr);
					minVals(0, j, k) = *temp;
					minIndices(0, j, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 2:

			minVals.create(nr, 1, nch);
			minIndices.create(nr, 1, nch);
			for (size_t k = 0; k < nch; k++)
			{
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					auto temp = std::min_element(ptr_cur, ptr_cur + nc);
					minVals(i, 0, k) = *temp;
					minIndices(i, 0, k) = std::distance(ptr_cur, temp);
				}
			}
			break;

		case 3:

			minVals.create(nr, nc, 1);
			minIndices.create(nr, nc, 1);
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					auto temp = std::min_element(ptr_cur, ptr_cur + nch);
					minVals(i, j) = *temp;
					minIndices(i, j) = std::distance(ptr_cur, temp);
				}
			break;
		}
	}

	// given a matrix, return a matrix that contains the median along a given dimension.
	// if dim==1, returns a row vector that contains median of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains median of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains median of each channel tube.
	// Usage is the same as Matlab's median() function. Gives same results also.
	Matkr<T> median(int dim)
	{

		// Implementation note:
		// uses std::nth_element which is partial sorting which is faster than sorting.
		// Finding the middle element by partial sorting. Then number of elements is odd,
		// I need to get the next element. This is done by finding the max value within
		// this partial sorted values (which are guaranteed to be less than the middle value
		// but in any order). This should still be faster than simply sorting all the elements.

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matkr<T> matOut;


		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);

			int medIdx = nr / 2;
			T vn;

			if (nr % 2 != 0) // when number of elements is odd (easy case)
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t j = 0; j < nc; j++)
					{
						Matkr<T> col_cur = gch(k).gc(j);
						T* ptr_cur = col_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nr);
						matOut(0, j, k) = ptr_cur[medIdx];
					}
			}

			else // when number of elements is even
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t j = 0; j < nc; j++)
					{
						Matkr<T> col_cur = gch(k).gc(j);
						T* ptr_cur = col_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nr);
						vn = ptr_cur[medIdx];
						matOut(0, j, k) = 0.5*(vn + *std::max_element(ptr_cur, ptr_cur + medIdx));
					}
			}
		}

			break;

		case 2:

		{
			matOut.create(nr, 1, nch);

			int medIdx = nc / 2;
			T vn;

			if (nc % 2 != 0) // when number of elements is odd (easy case)
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t i = 0; i < nr; i++)
					{
						Matkr<T> row_cur = gch(k).gr(i);
						T* ptr_cur = row_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nc);
						matOut(i, 0, k) = ptr_cur[medIdx];
					}
			}

			else // when number of elements is even
			{
				for (size_t k = 0; k < nch; k++)
					for (size_t i = 0; i < nr; i++)
					{
						Matkr<T> row_cur = gch(k).gr(i);
						T* ptr_cur = row_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nc);
						vn = ptr_cur[medIdx];
						matOut(i, 0, k) = 0.5*(vn + *std::max_element(ptr_cur, ptr_cur + medIdx));
					}
			}
		}

			break;

		case 3:

		{
			matOut.create(nr, nc, 1);

			int medIdx = nch / 2;
			T vn;

			if (nch % 2 != 0) // when number of elements is odd (easy case)
			{
				for (size_t j = 0; j < nc; j++)
					for (size_t i = 0; i < nr; i++)
					{
						Matkr<T> submat_cur = gs(i, i, j, j, 0, nch);
						T* ptr_cur = submat_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nch);
						matOut(i, j) = ptr_cur[medIdx];
					}
			}

			else // when number of elements is even
			{
				for (size_t j = 0; j < nc; j++)
					for (size_t i = 0; i < nr; i++)
					{
						Matkr<T> submat_cur = gs(i, i, j, j, 0, nch);
						T* ptr_cur = submat_cur.get_ptr();
						std::nth_element(ptr_cur, ptr_cur + medIdx, ptr_cur + nch);
						vn = ptr_cur[medIdx];
						matOut(i, j) = 0.5*(vn + *std::max_element(ptr_cur, ptr_cur + medIdx));
					}
			}	
		}
			
			break;
		}

		return matOut;
	}

	// given a matrix, return a matrix that contains the mean along a given dimension.
	// if dim==1, returns a row vector that contains mean of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains mean of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains mean of each channel tube.
	// Usage is the same as Matlab's mean() function. Gives same results also.
	Matkr<T> mean(int dim)
	{

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matkr<T> matOut;

		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);
			
			for (size_t k = 0; k < nch; k++)
				for (size_t j = 0; j < nc; j++)
				{
					Matkr<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					matOut(0, j, k) = std::accumulate(ptr_cur, ptr_cur+nr, 0.0) / (double)nr;
				}
		}

		break;

		case 2:

		{
			matOut.create(nr, 1, nch);

			for (size_t k = 0; k < nch; k++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					matOut(i, 0, k) = std::accumulate(ptr_cur, ptr_cur + nc, 0.0) / (double)nc;
				}
		}

		break;

		case 3:

		{
			matOut.create(nr, nc, 1);
	
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					matOut(i, j) = std::accumulate(ptr_cur, ptr_cur + nch, 0.0) / (double)nch;
				}
		}

		break;
		}

		return matOut;
	}

	// given a matrix, return a matrix that contains the sum along a given dimension.
	// if dim==1, returns a row vector that contains sum of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains sum of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains sum of each channel tube.
	// Usage is the same as Matlab's sum() function. Gives same results also.
	Matkr<T> sum(int dim)
	{

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matkr<T> matOut;

		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);

			for (size_t k = 0; k < nch; k++)
				for (size_t j = 0; j < nc; j++)
				{
					Matkr<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					matOut(0, j, k) = std::accumulate(ptr_cur, ptr_cur + nr, 0.0);
				}
		}

		break;

		case 2:

		{
			matOut.create(nr, 1, nch);

			for (size_t k = 0; k < nch; k++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					matOut(i, 0, k) = std::accumulate(ptr_cur, ptr_cur + nc, 0.0);
				}
		}

		break;

		case 3:

		{
			matOut.create(nr, nc, 1);

			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					matOut(i, j) = std::accumulate(ptr_cur, ptr_cur + nch, 0.0);
				}
		}

		break;
		}

		return matOut;
	}

	// given a matrix, return a matrix that contains the std/var along a given dimension.
	// if dim==1, returns a row vector that contains std/var of each column. If nchannels>1, process independently.
	// if dim==2, returns a col vector that contains std/var of each row. If nchannels>1, process independently.
	// if dim==3, returns a matrix that contains std/var of each channel tube.
	// Usage is similar to Matlab's std/var function. Gives same results.
	// norm_offset can be either 1 or 0. If 1, divide by N-1 (sample estimate). If 0, divide by N.
	// if var==true, computer variance instead of standard deviation
	Matkr<T> std(int dim, int norm_offset = 1, bool variance = false)
	{

		if ((norm_offset != 1) && (norm_offset != 0))
		{
			printf("norm_offset can only be 1 (division by N-1) or 0 (division by N). Using 1 as the norm_offset.\n");
			norm_offset = 1;
		}

		if ((dim != 1) && (dim != 2) && (dim != 3))
		{
			printf("dim can only be either 1, 2 or 3. Using 1 as the dim.\n");
			dim = 1;
		}

		Matkr<T> matOut;
		T mean_val, sq_sum;
		std::vector<T> diff;

		switch (dim)
		{
		case 1:

		{
			matOut.create(1, nc, nch);
			diff.resize(nr);

			for (size_t k = 0; k < nch; k++)
				for (size_t j = 0; j < nc; j++)
				{
					Matkr<T> col_cur = gch(k).gc(j);
					T* ptr_cur = col_cur.get_ptr();
					mean_val = std::accumulate(ptr_cur, ptr_cur + nr, 0.0) / (double)nr;
					std::transform(ptr_cur, ptr_cur + nr, diff.begin(), [mean_val](T x) { return x - mean_val; });
					sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
					matOut(0, j, k) = sq_sum / (nr - norm_offset);
				}
		}

		break;

		case 2:

		{
			matOut.create(nr, 1, nch);
			diff.resize(nc);

			for (size_t k = 0; k < nch; k++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> row_cur = gch(k).gr(i);
					T* ptr_cur = row_cur.get_ptr();
					mean_val = std::accumulate(ptr_cur, ptr_cur + nc, 0.0) / (double)nc;
					std::transform(ptr_cur, ptr_cur + nc, diff.begin(), [mean_val](T x) { return x - mean_val; });
					sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
					matOut(i, 0, k) = sq_sum / (nc - norm_offset);
				}
		}

		break;

		case 3:

		{
			matOut.create(nr, nc, 1);
			diff.resize(nch);

			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
				{
					Matkr<T> submat_cur = gs(i, i, j, j, 0, nch);
					T* ptr_cur = submat_cur.get_ptr();
					mean_val = std::accumulate(ptr_cur, ptr_cur + nch, 0.0) / (double)nch;
					std::transform(ptr_cur, ptr_cur + nch, diff.begin(), [mean_val](T x) { return x - mean_val; });
					sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
					matOut(i, j) = sq_sum / (nch - norm_offset);
				}
		}

		break;
		}

		if (!variance) // if std instead of variance requested, need to take square root
		{
			T* ptr_out = matOut.get_ptr();
			for (size_t i = 0; i < matOut.nelem(); i++)
				ptr_out[i] = std::sqrt(ptr_out[i]);
		}

		return matOut;
	}

	// computes histogram of the matrix which can be size and any nchannels.
	// the algorithm just treats the given matrix as a linear array (taken in column major).
	// edges is a col/row vector that contains the edges of the bins as follows:
	// {e1,e2,e3,...,eN}. The first bin is a count of values that fall in
	// within range [e1,e2), [e2,e3), ..., [eN-1,eN].
	// nbins = N - 1. Therefore, edges vector must have at least 2 elements.
	// if I want to count integers/catogories, then I can put a dummy last edge.
	// e.g. if given data Matkrd e = { 2, 4, 1, 2, 5, 2 }, computing the hist
	// using e.hist({ 1,2,3,4,5 }) gives [1,3,0,2] as output.
	// This function gives same results as matlab's histcounts(X,edges).
	// which bin to chose (to increment), rather than searching linearly,
	// I use std::upper_bound which has complexity of log(n) instead of n.
	Matkr<int> hist(const Matkr<T> &edges)
	{
		int nedges = edges.nelem();
		int nbins = nedges - 1;		
		Matkr<int> matOut(nbins, 1, 1);
		matOut.zeros();

		int *ptr_out = matOut.get_ptr();
		T *ptr_edges = edges.get_ptr();
		T *ptr_edges_end = edges.get_ptr() + nedges;

		int idx_edge_last = nedges - 1;
		T curVal;

		int idx_ub, idx_bin;

		for (size_t i = 0; i < ndata; i++)
		{
			curVal = ptr[i];			
			idx_ub = std::upper_bound(ptr_edges, ptr_edges_end, curVal) - ptr_edges;

			// handle boundary case (left most side)
			if (idx_ub == 0)
			{
				// data less than e1 (the first edge), so don't count
				if (curVal < ptr_edges[0])
					continue;				
			}					

			// handle boundary case (right most side)
			if (idx_ub == nedges)
			{
				// data greater than the last edge, so don't count
				if (curVal > ptr_edges[idx_edge_last])
					continue;
				// need to decrement since due to being at exactly edge final
				--idx_ub;					
			}

			idx_bin = idx_ub - 1;			
			++ptr_out[idx_bin];				
		}			

		return matOut;
	}

	// join this matrix with the given matrix matIn horizontally
	// if the number of rows or channels are different, max of them
	// will be taken and filled with zeros.
	Matkr<T> add_cols(const Matkr<T> &matIn)
	{
		int nrows_new = std::max(nr, matIn.nrows());
		int ncols_new = nc + matIn.ncols();
		int nch_new = std::max(nch, matIn.nchannels());
		Matkr<T>matOut(nrows_new, ncols_new, nch_new);
		matOut.ss(0, nr - 1, 0, nc - 1, 0, nch - 1) = *this;
		matOut.ss(0, matIn.nrows() - 1, nc, ncols_new - 1, 0, matIn.nchannels() - 1) = matIn;
		return matOut;
	}

	// merge a std::vector of matrices horizontally
	// if the number of rows or channels are different, max of them
	// will be taken and filled with zeros.
	static Matkr<T> merge_cols(const std::vector<Matkr<T>> &vmat)
	{
		int nrows_new = 0;
		int ncols_new = 0;
		int nch_new = 0;
		int nmats = vmat.size();

		for (size_t kk = 0; kk < nmats; kk++)
		{
			nrows_new = std::max(nrows_new, vmat[kk].nrows());
			ncols_new += vmat[kk].ncols();
			nch_new = std::max(nch_new, vmat[kk].nchannels());
		}
 
		Matkr<T>matOut(nrows_new, ncols_new, nch_new);
		int nc_count = 0;

		for (size_t kk = 0; kk < nmats; kk++)
		{
			matOut.ss(0, vmat[kk].nrows() - 1, nc_count, nc_count + vmat[kk].ncols() - 1, 0, vmat[kk].nchannels() - 1) = vmat[kk];
			nc_count += vmat[kk].ncols();
		}

		return matOut;
	}

	// join this matrix with the given matrix matIn vertically
	// if the number of cols or channels are different, max of them
	// will be taken and filled with zeros.
	Matkr<T> add_rows(const Matkr<T> &matIn)
	{		
		int nrows_new = nr + matIn.nrows();
		int ncols_new = std::max(nc, matIn.ncols());
		int nch_new = std::max(nch, matIn.nchannels());
		Matkr<T>matOut(nrows_new, ncols_new, nch_new);
		matOut.ss(0, nr - 1, 0, nc - 1, 0, nch - 1) = *this;
		matOut.ss(nr, nrows_new - 1, 0, matIn.ncols() - 1, 0, matIn.nchannels() - 1) = matIn;
		return matOut;
	}

	// merge a std::vector of matrices vertically
	// if the number of cols or channels are different, max of them
	// will be taken and filled with zeros.
	static Matkr<T> merge_rows(const std::vector<Matkr<T>> &vmat)
	{
		int nrows_new = 0;
		int ncols_new = 0;
		int nch_new = 0;
		int nmats = vmat.size();

		for (size_t kk = 0; kk < nmats; kk++)
		{			
			nrows_new += vmat[kk].nrows();
			ncols_new = std::max(ncols_new, vmat[kk].ncols());
			nch_new = std::max(nch_new, vmat[kk].nchannels());
		}

		Matkr<T>matOut(nrows_new, ncols_new, nch_new);
		int nr_count = 0;

		for (size_t kk = 0; kk < nmats; kk++)
		{
			matOut.ss(nr_count, nr_count + vmat[kk].nrows() - 1, 0, vmat[kk].ncols() - 1, 0, vmat[kk].nchannels() - 1) = vmat[kk];
			nr_count += vmat[kk].nrows();
		}

		return matOut;
	}

	// add channels to the current matrix.
	// if rows and columns of the two matrices are different, max of them
	// will be taken and filled with zeros
	Matkr<T> add_channels(const Matkr<T> &matIn)
	{
		int nrows_new = std::max(nr, matIn.nrows());
		int ncols_new = std::max(nc, matIn.ncols());
		int nch_new = nch + matIn.nchannels();
		Matkr<T>matOut(nrows_new, ncols_new, nch_new);
		matOut.ss(0, nr - 1, 0, nc - 1, 0, nch - 1) = *this;
		matOut.ss(0, matIn.nrows() - 1, 0, matIn.ncols() - 1, nch, nch_new - 1) = matIn;
		return matOut;
	}

	// merge channels of a std::vector of matrices
	// if rows and columns of the two matrices are different, max of them
	// will be taken and filled with zeros
	static Matkr<T> merge_channels(const std::vector<Matkr<T>> &vmat)
	{
		int nrows_new = 0;
		int ncols_new = 0;
		int nch_new = 0;
		int nmats = vmat.size();

		for (size_t kk = 0; kk < nmats; kk++)
		{
			nrows_new = std::max(nrows_new, vmat[kk].nrows());
			ncols_new = std::max(ncols_new, vmat[kk].ncols());
			nch_new += vmat[kk].nchannels();
		}

		Matkr<T>matOut(nrows_new, ncols_new, nch_new);
		int nch_count = 0;

		for (size_t kk = 0; kk < nmats; kk++)
		{
			matOut.ss(0, vmat[kk].nrows() - 1, 0, vmat[kk].ncols() - 1, nch_count, nch_count + vmat[kk].nchannels() - 1) = vmat[kk];
			nch_count += vmat[kk].nchannels();
		}

		return matOut;
	}

	// remove cols from matrix
	Matkr<T> del_cols(Veck<int> indices_remove)
	{		
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// row indices that I want to keep (keep all; 1,2,...,nr)
		Veck<int> row_idxs_keep = Veck<int>::fill_ladder(nr, 0, 1);

		// channel indices that I want to keep (keep all; 1,2,...,nch)
		Veck<int> ch_idxs_keep = Veck<int>::fill_ladder(nch, 0, 1);

		// col indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> col_idxs_keep;
		// create a vector that is filled with 0,1,2,...,nc
		Veck<int> col_idxs_all = Veck<int>::fill_ladder(nc, 0, 1);
		std::set_difference(col_idxs_all.begin(), col_idxs_all.end(), 
			indices_remove.begin(), indices_remove.end(),
			std::inserter(col_idxs_keep, col_idxs_keep.begin()));

		Matkr<T>matOut = gs(row_idxs_keep, Veck<int>(col_idxs_keep), ch_idxs_keep);
		
		return matOut;
	}

	// remove rows from matrix
	Matkr<T> del_rows(Veck<int> indices_remove)
	{
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// col indices that I want to keep (keep all; 1,2,...,nc)
		Veck<int> col_idxs_keep = Veck<int>::fill_ladder(nc, 0, 1);

		// channel indices that I want to keep (keep all; 1,2,...,nch)
		Veck<int> ch_idxs_keep = Veck<int>::fill_ladder(nch, 0, 1);

		// row indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> row_idxs_keep;
		// create a vector that is filled with 0,1,2,...,nr
		Veck<int> row_idxs_all = Veck<int>::fill_ladder(nr, 0, 1);
		std::set_difference(row_idxs_all.begin(), row_idxs_all.end(),
			indices_remove.begin(), indices_remove.end(),
			std::inserter(row_idxs_keep, row_idxs_keep.begin()));

		Matkr<T>matOut = gs(Veck<int>(row_idxs_keep), col_idxs_keep, ch_idxs_keep);

		return matOut;
	}

	// remove channels from matrix
	Matkr<T> del_channels(Veck<int> indices_remove)
	{
		// make sure that given indices are sorted in ascending order
		std::sort(indices_remove.begin(), indices_remove.end());

		// row indices that I want to keep (keep all; 1,2,...,nr)
		Veck<int> row_idxs_keep = Veck<int>::fill_ladder(nr, 0, 1);

		// col indices that I want to keep (keep all; 1,2,...,nc)
		Veck<int> col_idxs_keep = Veck<int>::fill_ladder(nc, 0, 1);

		// channel indices that I want to keep (keep the ones not meant to be removed)
		std::vector<int> ch_idxs_keep;
		// create a vector that is filled with 0,1,2,...,nch
		Veck<int> ch_idxs_all = Veck<int>::fill_ladder(nch, 0, 1);
		std::set_difference(ch_idxs_all.begin(), ch_idxs_all.end(),
			indices_remove.begin(), indices_remove.end(),
			std::inserter(ch_idxs_keep, ch_idxs_keep.begin()));

		Matkr<T>matOut = gs(row_idxs_keep, col_idxs_keep, Veck<int>(ch_idxs_keep));

		return matOut;
	}
	
				
	int nrows() const { return nr; }
	int ncols() const { return nc; }
	int nchannels() const { return nch; }
	int nelem() const { return ndata; }
	T* get_ptr() const { return ptr; }
	T* begin() const { return ptr; } // to ease use in STL containers/algs
	T* end() const { return ptr + ndata; } // to ease use in STL containers/algs
		
protected:
	std::unique_ptr<T[]> data;
	T *ptr; // pointer to data;
	int nr;
	int nc;
	int nch;
	int ndata; // nr*nc*nch

	void init(int nrows, int ncols, int nchannels)
	{
		nr = nrows; nc = ncols; nch = nchannels;
		ndata = nr * nc * nch;
		data = std::make_unique<T[]>(ndata);
		ptr = data.get();
	}

	void init(int nrows, int ncols, int nchannels, const T* data_external, bool external_is_row_major=true)
	{
		init(nrows, ncols, nchannels);
		if(external_is_row_major)
			std::memcpy(ptr, data_external, sizeof(T)*ndata);
		else
		{
			unsigned int counter = 0;
			for (size_t k = 0; k < nchannels; k++)
				for (size_t j = 0; j < ncols; j++)
					for (size_t i = 0; i < nrows; i++)	
						(*this)(i,j,k) = data_external[counter++];
		}
	}

};

typedef Matkr<double> Matkrd;
typedef Matkr<float> Matkrf;
typedef Matkr<int> Matkri;
typedef Matkr<short> Matkrs;
typedef Matkr<unsigned char> Matkruc;
typedef Matkr<long> Matkrl;

typedef Veck<double> veckd;
typedef Veck<float> veckf;
typedef Veck<int> vecki;
typedef Veck<short> vecks;
typedef Veck<unsigned char> veckuc;
typedef Veck<long> veckl;

#endif




/*

Code usage examples (only cover a very small of operations; for others, refer to the library directly):

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
	74.0000 19.0000 2.0000  61.0000 9.0000
	53.0000 13.0000 90.0000 77.0000 74.0000
	=========================
	Channel 2:
	34.0000 18.0000 69.0000 98.0000 33.0000
	84.0000 13.0000 74.0000 40.0000 32.0000
	38.0000 88.0000 44.0000 45.0000 90.0000
	83.0000 5.0000  38.0000 16.0000 25.0000
	=========================

	Channel 1:
	52.0000 83.0000 86.0000
	37.0000 64.0000 39.0000
	74.0000 2.0000  9.0000
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
	74.0000 2.0000  9.0000
	53.0000 90.0000 74.0000
	52.0000 83.0000 86.0000
	37.0000 64.0000 39.0000
	=========================
	Channel 2:
	0.0000  0.0000  0.0000
	0.0000  0.0000  0.0000
	0.0000  0.0000  0.0000
	0.0000  0.0000  0.0000
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
	74.0000 2.0000  9.0000
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
	2.0000  9.0000
	90.0000 74.0000
	
Code  (cont'd):

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
	99.5000 45.0000 3.0000  40.0000	
	

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
	74.0000 19.0000 2.0000  61.0000 9.0000
	53.0000 13.0000 90.0000 77.0000 74.0000
	=========================
	Channel 2:
	34.0000 18.0000 69.0000 98.0000 33.0000
	84.0000 13.0000 74.0000 40.0000 32.0000
	38.0000 88.0000 44.0000 45.0000 90.0000
	83.0000 5.0000  38.0000 16.0000 25.0000
	=========================
	
	after:

	Channel 1:
	52.0000 81.0000 1.0000  3.0000  86.0000
	37.0000 82.0000 2.0000  4.0000  39.0000
	74.0000 19.0000 2.0000  61.0000 9.0000
	53.0000 13.0000 90.0000 77.0000 74.0000
	=========================
	Channel 2:
	34.0000 18.0000 69.0000 98.0000 33.0000
	84.0000 13.0000 74.0000 40.0000 32.0000
	38.0000 88.0000 44.0000 45.0000 90.0000
	83.0000 5.0000  38.0000 16.0000 25.0000
	=========================


*/

