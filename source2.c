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

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "header.h"

int main(int argc, char *argv[]) {
  int m, n, p;

  m = 16;
  printf("Matrix size: %d^2\n", m);

  // init square matrix
  p = n = m;

  // leading dimension
  int lda = m;
  int ldb = n;
  int ldc = m;
  int begintime, endtime;

  // memory allocation
  fdata *a = ALLOC_ARR(m * lda);
  double *ad = ALLOC_ARRD(m * lda);
  fdata *b = ALLOC_ARR(n * ldb);
  double *bd = ALLOC_ARRD(n * ldb);
  fdata *res = ALLOC_ARR(m * ldc);
  fdata *res0 = ALLOC_ARR(m * ldc);
  fdata *res1 = ALLOC_ARR(m * ldc);
  fdata *res2 = ALLOC_ARR(m * ldc);
  double *res3 = ALLOC_ARRD(m * ldc);
  fdata *ref = ALLOC_ARR(m * ldc);
  struct timeval start, end;

  rand_matrix(a, m, p, lda);
  for (int count = 0; count < m * p; count++) {
    ad[count] = a[count];
  }
  rand_matrix(b, p, n, ldb);
  for (int count = 0; count < n * p; count++) {
    bd[count] = a[count];
  }

  begintime = clock();
  // naive gemm
  gettimeofday(&start, NULL);
  matmul_native(m, n, p, a, lda, b, ldb, ref, ldc);
  gettimeofday(&end, NULL);
  endtime = clock();
  printf("Trivial: %.6f sec\n",
         ((end.tv_sec - start.tv_sec) * 1.0e6 + end.tv_usec - start.tv_usec) /
             1.0e6);
  printf("Running Time: %dms\n", endtime - begintime);

  begintime = clock();
  // naive gemm with openmp
  gettimeofday(&start, NULL);
  matmul_native_with_omp(m, n, p, a, lda, b, ldb, res, ldc);
  gettimeofday(&end, NULL);
  endtime = clock();
  printf("OpenMP: %.6f sec\n",
         ((end.tv_sec - start.tv_sec) * 1.0e6 + end.tv_usec - start.tv_usec) /
             1.0e6);
  printf("Running Time: %dms\n", endtime - begintime);
  float err = matrix_diff(res, ref, m, n, lda);
  printf("max elem-wise error = %g\n", err);
  if (err > 1e-5) {
    puts("ERROR");
  } else {
    puts("CORRECT");
  }

  begintime = clock();
  // cache hit optimized
  gettimeofday(&start, NULL);
  matmul_cache_opt(m, n, p, a, lda, b, ldb, res0, ldc);
  gettimeofday(&end, NULL);
  endtime = clock();
  printf("CacheOpt: %.6f sec\n",
         ((end.tv_sec - start.tv_sec) * 1.0e6 + end.tv_usec - start.tv_usec) /
             1.0e6);
  printf("Running Time: %dms\n", endtime - begintime);
  err = matrix_diff(res0, ref, m, n, lda);
  printf("max elem-wise error = %g\n", err);
  if (err > 1e-5) {
    puts("ERROR");
  } else {
    puts("CORRECT");
  }

  begintime = clock();
  // divide and conquer method
  gettimeofday(&start, NULL);
  matmul_dc_opt(m, n, p, a, lda, b, ldb, res1, ldc);
  gettimeofday(&end, NULL);
  endtime = clock();
  printf("DivConq: %.6f sec\n",
         ((end.tv_sec - start.tv_sec) * 1.0e6 + end.tv_usec - start.tv_usec) /
             1.0e6);
  printf("Running Time: %dms\n", endtime - begintime);
  err = matrix_diff(res1, ref, m, n, lda);
  printf("max elem-wise error = %g\n", err);
  if (err > 1e-5) {
    puts("ERROR");
  } else {
    puts("CORRECT");
  }

  begintime = clock();
  // strassen
  gettimeofday(&start, NULL);
  matmul_strassen(m, n, p, a, lda, b, ldb, res2, ldc);
  gettimeofday(&end, NULL);
  endtime = clock();
  printf("Strassen: %.6f sec\n",
         ((end.tv_sec - start.tv_sec) * 1.0e6 + end.tv_usec - start.tv_usec) /
             1.0e6);
  printf("Running Time: %dms\n", endtime - begintime);
  err = matrix_diff(res2, ref, m, n, lda);
  printf("max elem-wise error = %g\n", err);
  if (err > 1e-5) {
    puts("ERROR");
  } else {
    puts("CORRECT");
  }

  begintime = clock();
  // openblas lib
  gettimeofday(&start, NULL);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, 1.0f, ad, lda,
              bd, ldb, 1.0f, res3, ldc);
  gettimeofday(&end, NULL);
  endtime = clock();
  printf("OpenBlas: %.6f sec\n",
         ((end.tv_sec - start.tv_sec) * 1.0e6 + end.tv_usec - start.tv_usec) /
             1.0e6);
  printf("Running Time: %dms\n", endtime - begintime);
  err = matrix_diff(res2, ref, m, n, lda);
  printf("max elem-wise error = %g\n", err);
  if (err > 1e-5) {
    puts("ERROR");
  } else {
    puts("CORRECT");
  }

  free(a);
  free(b);
  free(ad);
  free(bd);
  free(res);
  free(res0);
  free(res1);
  free(res2);
  free(res3);
  free(ref);

  return EXIT_SUCCESS;
}