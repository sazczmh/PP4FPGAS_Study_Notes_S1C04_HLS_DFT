typedef float DTYPE;
#define SIZE 256							// DFT Size

//#define S1_Baseline
#define S2_SPipeline
//#define S3_Loop_Interchange
//#define S4_LUT
//#define S5_Manual_Unroll
void dft(DTYPE sample_real[SIZE], DTYPE sample_imag[SIZE]);


