/*
 Copyright (c) 2022 SUSTech - JustLittleFive

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <immintrin.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef float fdata;

// allocate fdata with alignment to optimize memory and cache
#define ALLOC_ARR(n) \
  (fdata*)calloc_align(sizeof(fdata) * 4, (n) * sizeof(fdata))

#define ALLOC_ARRD(n) \
  (double*)calloc_align(sizeof(double) * 4, (n) * sizeof(double))

void* calloc_align(size_t, size_t);

void rand_matrix(fdata*, int, int, int);

void matmul_native(int, int, int, const fdata*, int, const fdata*, int, fdata*,
                   int);
void matmul_native_with_omp(int, int, int, const fdata*, int, const fdata*, int,
                            fdata*, int);
void matmul_cache_opt(int, int, int, const fdata*, int, const fdata*, int,
                      fdata*, int);
void matmul_dc_opt(int, int, int, const fdata*, int, const fdata*, int, fdata*,
                   int);
void matmul_strassen(int, int, int, const fdata*, int, const fdata*, int,
                     fdata*, int);
float matrix_diff(fdata*, fdata*, int, int, int);
