#include <iostream>
#include <fstream>
#include <vector>
#include "mmio.h"

#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#ifdef defined(__APPLE__) || defined(MACOS)
	#include <OpenCL/cl.hpp>
#else
	#include <CL/cl.hpp>
#endif

/*** Config ***/
#define MATRIX_FILE "../matrix.mtx"

int main() {
	/*** Data objects ***/
	MM_typecode code;
	int32_t num_rows, num_values;
	int32_t *rows = nullptr;
	int32_t *cols = nullptr;
	float *values = nullptr;
	float *b = nullptr;
	float *x = nullptr;

	/*** OpenCL objects ***/
	cl::Platform platform;
	std::vector<cl::Device> devices;
	cl::Device device;
	cl::Context context;
	std::string source;
	cl::Program program;
	cl::Kernel kernel;
	cl::Buffer rows_buffer, cols_buffer, values_buffer, b_buffer, x_buffer;
	cl::CommandQueue queue;

	try {

		std::cout << "Start reading kernel...";
		/*** Read kernel source file ***/
		std::ifstream kernelFile("../kernel.cl");
		if (!kernelFile)
			throw std::runtime_error("Kernel file not found");

		std::string line;
		while (std::getline(kernelFile, line))
			source += line + '\n';

		kernelFile.close();
		std::cout << "[ok]" << std::endl;

		std::cout << "Start reading matrix...";
		/*** Read sparse matrix ***/
		FILE *matrix_handle = fopen(MATRIX_FILE, "r");
		if (matrix_handle == NULL)
			throw std::runtime_error("Matrix file not found");

		/*** Read matrix file parameters ***/
		mm_read_banner(matrix_handle, &code);
		mm_read_mtx_crd_size(matrix_handle, &num_rows, &num_rows, &num_values);
		if (mm_is_symmetric(code) || mm_is_skew(code) || mm_is_hermitian(code))
			num_values += num_values - num_rows;

		/*** Allocate memory ***/
		rows = new int32_t [num_values];
		cols = new int32_t [num_values];
		values = new float [num_values];
		b = new float [num_values];
		x = new float [num_values];

		/*** Read numbers from matrix file ***/
		for (size_t i=0; i<num_values; i++) {
			int32_t row, col;
			double value;
			fscanf(matrix_handle, "%d %d %lg\n", &row, &col, &value);
			rows[i] = --row;
			cols[i] = --col;
			values[i] = (float)value;

			if ((rows[i] != cols[i]) && (mm_is_symmetric(code) || mm_is_skew(code) || mm_is_hermitian(code))) {
				i++;
				rows[i] = cols[i-1];
				cols[i] = rows[i-1];
				values[i] = values[i-1];
			}
		}

		fclose(matrix_handle);

		/*** Sort matrix by cols ***/
		int32_t index = 0;
		for (size_t i=0; i<num_values; i++) {
			for (size_t j=0; j<num_values; j++) {
				if (rows[j] == i) {
					if (j == index)
						index++;

					else if (j > index) {
						std::swap(rows[j], rows[index]);
						std::swap(cols[j], cols[index]);
						std::swap(values[j], values[index]);

						index++;
					}
				}
			}
		}
		std::cout << "[ok]" << std::endl;

		/*** Generate b vector ***/
		for (size_t i=0; i<num_rows; i++)
			b[i] = (i+1) * 100000000;

		/*** Get default OpenCL platform ***/
		platform = cl::Platform::getDefault();
		std::cout << "\t" << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

		/*** Get all platform devices and select first ***/
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		device = devices[0];
		std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;

		/*** Create context and build kernel program ***/
		context = cl::Context({ device });
		program = cl::Program(context, source);
		program.build();
		kernel = cl::Kernel(program, "solver");

		/*** Create buffers and upload data ***/
		rows_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
								 num_values * sizeof(int32_t), rows);

		cols_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		                         num_values * sizeof(int32_t), cols);

		values_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		                         num_values * sizeof(float), values);

		b_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		                         num_rows * sizeof(float), b);

		x_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, num_rows * sizeof(float),
							  NULL);

		/*** Set OpenCL kernel arguments ***/
		kernel.setArg(0, sizeof(int32_t), &num_rows);
		kernel.setArg(1, sizeof(int32_t), &num_values);
		kernel.setArg(2, rows_buffer);
		kernel.setArg(3, cols_buffer);
		kernel.setArg(4, values_buffer);
		kernel.setArg(5, b_buffer);
		kernel.setArg(6, x_buffer);
		kernel.setArg(7, num_values * sizeof(float), NULL);
		kernel.setArg(8, num_values * sizeof(float), NULL);
		kernel.setArg(9, num_values * sizeof(float), NULL);

		/*** Create commandQueue and run kernel ***/
		queue = cl::CommandQueue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_rows), cl::NullRange);
		queue.finish();

		/*** Read x buffer data ***/
		queue.enqueueReadBuffer(x_buffer, CL_TRUE, 0, num_rows * sizeof(float),
								x, NULL, NULL);

		std::cout << std::endl << "Solve:" << std::endl;
		for (size_t i=0; i<num_rows; i++)
			std::cout << i+1 << ": " << x[i] << std::endl;

	} catch (cl::Error e) {
		std::cerr << std::endl << e.what() << ": Error code " << e.err() << std::endl;
	} catch (std::runtime_error e) {
		std::cerr << std::endl << e.what() << std::endl;
	} catch (...) {
		std::cerr << std::endl << "Unknow error" << std::endl;
	}

	/*** Delete allocated memory ***/
	delete [] rows;
	delete [] cols;
	delete [] values;
	delete [] b;
	delete [] x;

	return EXIT_SUCCESS;
}
