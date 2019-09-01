#include"matrix_vector_base.h"


//*********************S1_Baseline
#ifdef S1_Baseline
void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j++) {
			sum += V_In[j] * M[i][j];
		}
		V_Out[i] = sum;
	}
}

#endif


//*********************S2_Manual_UNROLL
#ifdef S2_Manual_UNROLL
void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
		BaseType sum = 0;
		V_Out[i] =	V_In[0] * M[i][0] + V_In[1] * M[i][1] + V_In[2] * M[i][2] +
					V_In[3] * M[i][3] + V_In[4] * M[i][4] + V_In[5] * M[i][5] +
					V_In[6] * M[i][6] + V_In[7] * M[i][7];
	}
}

#endif


//*********************S3_Auto_UNROLL
#ifdef S3_Auto_UNROLL

void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {

	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j++) {
#pragma HLS UNROLL skip_exit_check
			sum += V_In[j] * M[i][j];
		}
		V_Out[i] = sum;
	}
}

#endif


//*********************S4_ARRAY_PARTITION
#ifdef S4_ARRAY_PARTITION

void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=M complete dim=2
#pragma HLS ARRAY_PARTITION variable=V_In complete dim=1

	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j++) {
#pragma HLS UNROLL skip_exit_check
			sum += V_In[j] * M[i][j];
		}
		V_Out[i] = sum;
	}
}

#endif


//*********************S4_ARRAY_PARTITION_4
#ifdef S4_ARRAY_PARTITION_4

void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=M cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=V_In cyclic factor=4 dim=1

	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j++) {
#pragma HLS UNROLL skip_exit_check
			sum += V_In[j] * M[i][j];
		}
		V_Out[i] = sum;
	}
}

#endif

//*********************S5_PIPELINE
#ifdef S5_PIPELINE

void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=M complete dim=2
#pragma HLS ARRAY_PARTITION variable=V_In complete dim=1

	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j++) {
#pragma HLS UNROLL skip_exit_check
			sum += V_In[j] * M[i][j];
		}
		V_Out[i] = sum;
	}
}

#endif


//*********************S6_Unit_PIPELINE
#ifdef S6_Unit_PIPELINE

void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=M cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=V_In cyclic factor=4 dim=1

	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE II=2
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j++) {
			sum += V_In[j] * M[i][j];
		}
		V_Out[i] = sum;
	}
}

#endif


//*********************S7_Test
#ifdef S7_Test

void matrix_vector(BaseType M[SIZE][SIZE], BaseType V_In[SIZE], BaseType V_Out[SIZE]) {
#pragma HLS RESOURCE variable=V_In core=RAM_1P
//#pragma HLS array_partition variable=M dim=2 cyclic factor=2
//#pragma HLS array_partition variable=V_In cyclic factor=2
	BaseType i, j;
data_loop:
	for (i = 0; i < SIZE; i++) {
		BaseType sum = 0;
	dot_product_loop:
		for (j = 0; j < SIZE; j+=2) {
#pragma HLS pipeline II=1
			sum += V_In[j] * M[i][j];
			sum += V_In[j+1] * M[i][j+1];
		}
		V_Out[i] = sum;
	}
}

#endif
