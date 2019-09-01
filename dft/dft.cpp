#include "dft.h"
#include <math.h>					//Required for cos and sin functions


#ifdef S1_Baseline
//*****************S1_Baseline
void dft(DTYPE sample_real[SIZE], DTYPE sample_imag[SIZE]) {
	int i, j;
	DTYPE w;
	DTYPE c, s;
	// Temporary arrays to hold the intermediate frequency domain results
	DTYPE temp_real[SIZE];
	DTYPE temp_imag[SIZE];
	// Calculate each frequency domain sample iteratively
	dft_each_Calculate:
	for (i = 0; i < SIZE; i += 1) {
		temp_real[i] = 0;
		temp_imag[i] = 0;
		// (2 * pi * i)/N
		w = (-2.0 * 3.141592653589  / SIZE) * (DTYPE)i;
		// Calculate the jth frequency sample sequentially
		dft_jthCalculate:
		for (j = 0; j < SIZE; j += 1) {
			// Utilize HLS tool to calculate sine and cosine values
			c = cos(j * w);
			s = sin(j * w);
			// Multiply the current phasor with the appropriate input sample and keep
			// running sum
			temp_real[i] += (sample_real[j] * c - sample_imag[j] * s);
			temp_imag[i] += (sample_real[j] * s + sample_imag[j] * c);
		}
	}
	// Perform an inplace DFT, i.e., copy result into the input arrays
	ARRAY_Copy:
	for (i = 0; i < SIZE; i += 1) {
		sample_real[i] = temp_real[i];
		sample_imag[i] = temp_imag[i];
	}
}
#endif


#ifdef S2_SPipeline
//*****************S2_SPipeline
void dft(DTYPE sample_real[SIZE], DTYPE sample_imag[SIZE]) {
	int i, j;
	DTYPE w;
	DTYPE c, s;
	// Temporary arrays to hold the intermediate frequency domain results
	DTYPE temp_real[SIZE];
	DTYPE temp_imag[SIZE];
	// Calculate each frequency domain sample iteratively
	dft_each_Calculate:
	for (i = 0; i < SIZE; i += 1) {
		temp_real[i] = 0;
		temp_imag[i] = 0;
		// (2 * pi * i)/N
		w = (-2.0 * 3.141592653589  / SIZE) * (DTYPE)i;
		// Calculate the jth frequency sample sequentially
		dft_jthCalculate:
		for (j = 0; j < SIZE; j += 1) {
#pragma HLS PIPELINE
			// Utilize HLS tool to calculate sine and cosine values
			c = cos(j * w);
			s = sin(j * w);
			// Multiply the current phasor with the appropriate input sample and keep
			// running sum
			temp_real[i] += (sample_real[j] * c - sample_imag[j] * s);
			temp_imag[i] += (sample_real[j] * s + sample_imag[j] * c);
		}
	}
	// Perform an inplace DFT, i.e., copy result into the input arrays
	ARRAY_Copy:
	for (i = 0; i < SIZE; i += 1) {
		sample_real[i] = temp_real[i];
		sample_imag[i] = temp_imag[i];
	}
}
#endif


#ifdef S3_Loop_Interchange
//*****************S3_Loop_Interchange
void dft(DTYPE sample_real[SIZE], DTYPE sample_imag[SIZE]) {
	int i, j;
	DTYPE w;
	DTYPE c, s;
	// Temporary arrays to hold the intermediate frequency domain results
	DTYPE temp_real[SIZE]={0};
	DTYPE temp_imag[SIZE]={0};
	// Calculate the jth frequency sample sequentially
	dft_jthCalculate:
	for (j = 0; j < SIZE; j += 1) {
		// (2 * pi * i)/N
		w = (-2.0 * 3.141592653589  / SIZE) * (DTYPE)j;
		// Calculate each frequency domain sample iteratively
		dft_each_Calculate:
		for (i = 0; i < SIZE; i += 1) {
#pragma HLS PIPELINE II=1
			// Utilize HLS tool to calculate sine and cosine values
			c = cos(i * w);
			s = sin(i * w);
			// Multiply the current phasor with the appropriate input sample and keep
			// running sum
			temp_real[i] += (sample_real[j] * c - sample_imag[j] * s);
			temp_imag[i] += (sample_real[j] * s + sample_imag[j] * c);
		}
	}
	// Perform an inplace DFT, i.e., copy result into the input arrays
	ARRAY_Copy:
	for (i = 0; i < SIZE; i += 1) {
#pragma HLS PIPELINE II=1
		sample_real[i] = temp_real[i];
		sample_imag[i] = temp_imag[i];
	}
}
#endif


#ifdef S4_LUT
//*****************S4_LUT
#include"coefficients256.h"
#include "ap_int.h"

void dft(DTYPE sample_real[SIZE], DTYPE sample_imag[SIZE]) {
	int i, j;
	DTYPE w;
	DTYPE c, s;
	// Temporary arrays to hold the intermediate frequency domain results
	DTYPE temp_real[SIZE]={0};
	DTYPE temp_imag[SIZE]={0};
	// Calculate the jth frequency sample sequentially
	dft_jthCalculate:
	for (j = 0; j < SIZE; j += 1) {
		// Calculate each frequency domain sample iteratively
		dft_each_Calculate:
		for (i = 0; i < SIZE; i += 1) {
#pragma HLS PIPELINE II=1
			// Utilize HLS tool to calculate sine and cosine values
			c = cos_coefficients_table[(ap_uint<8>)(i * j)];
			s = sin_coefficients_table[(ap_uint<8>)(i * j)];
			// Multiply the current phasor with the appropriate input sample and keep
			// running sum
			temp_real[i] += (sample_real[j] * c - sample_imag[j] * s);
			temp_imag[i] += (sample_real[j] * s + sample_imag[j] * c);
		}
	}
	// Perform an inplace DFT, i.e., copy result into the input arrays
	ARRAY_Copy:
	for (i = 0; i < SIZE; i += 1) {
#pragma HLS PIPELINE II=1
		sample_real[i] = temp_real[i];
		sample_imag[i] = temp_imag[i];
	}
}
#endif


#ifdef S5_Manual_Unroll
//*****************S5_Manual_Unroll
#include"coefficients256.h"
#include "ap_int.h"

void dft(DTYPE sample_real[SIZE], DTYPE sample_imag[SIZE]) {
	int i, j;
	DTYPE c_0, s_0;
	DTYPE c_1, s_1;
	// Temporary arrays to hold the intermediate frequency domain results
	DTYPE temp_real[SIZE]={0};
	DTYPE temp_imag[SIZE]={0};
	// Calculate the jth frequency sample sequentially
	dft_jthCalculate:
	for (j = 0; j < SIZE; j += 2) {
		// Calculate each frequency domain sample iteratively
		dft_each_Calculate:
		for (i = 0; i < SIZE; i += 1) {
#pragma HLS PIPELINE II=1
			// Utilize HLS tool to calculate sine and cosine values
			c_0 = cos_coefficients_table[(ap_uint<8>)(i * j)];
			s_0 = sin_coefficients_table[(ap_uint<8>)(i * j)];
			c_1 = cos_coefficients_table[(ap_uint<8>)((i) * (j+1))];
			s_1 = sin_coefficients_table[(ap_uint<8>)((i) * (j+1))];
			// Multiply the current phasor with the appropriate input sample and keep
			// running sum
			temp_real[i] += (sample_real[j] * c_0 - sample_imag[j] * s_0) +
							(sample_real[j + 1] * c_1 - sample_imag[j + 1] * s_1);
			temp_imag[i] += (sample_real[j] * s_0 + sample_imag[j] * c_0) +
							(sample_real[j + 1] * s_1 + sample_imag[j + 1] * c_1);
		}
	}
	// Perform an inplace DFT, i.e., copy result into the input arrays
	ARRAY_Copy:
	for (i = 0; i < SIZE; i += 1) {
#pragma HLS PIPELINE II=1
		sample_real[i] = temp_real[i];
		sample_imag[i] = temp_imag[i];
	}
}
#endif
