#include <iostream>
#include <cmath> // for abs function
#include <string>
#include <iomanip>
#include <cstdint>
 
 

#include <cfloat>
using namespace std;
 

struct Model
{
    short rows = 1, cols = 1, channels = 1, padding = 1, stride = 1, size = rows + 2 * padding;
};
//
void Convolution(float *input, short channels, short input_size, const float *filter, short filter_rows, short no_filters, float *output, short layer_size, short output_rows, const float *mean, const float *variance, const float *gamma, const float *beta)
{
    for (short a = 0; a < no_filters; a++)
    {
        float sum = 0.0f;

        for (short i = 0; i < channels; i++)
        {
            for (short k = 0; k < filter_rows; k++)
            {
                for (short l = 0; l < filter_rows; l++)
                {
                    float elm1 = input[i * filter_rows * input_size + k * filter_rows + l];
                    float elm2 = filter[a * channels * filter_rows * filter_rows + i * filter_rows * filter_rows + k * filter_rows + l];
                    sum += elm1 * elm2;
                    // cout<<elm1<<" "<<elm2<<endl;
                }
            }
        }
        sum = gamma[a] * ((sum - mean[a]) / sqrt(variance[a] + 1e-4f)) + beta[a];
        sum = (sum > 0) ? sum : 0;

        output[a * output_rows * layer_size] = sum;
    }
}

void MaxPooling(float *input, short channels, short input_size, short pool_size, float *output, short output_size, short output_rows)
{

    for (short i = 0; i < channels; i++)
    {
        float max_val = -FLT_MAX; // Start with the smallest possible value (negative infinity)

        for (short k = 0; k < pool_size; k++)
        {
            for (short l = 0; l < pool_size; l++)
            {
                float val = input[(i * input_size * pool_size) + (k * pool_size) + l];
                if (val > max_val)
                {
                    max_val = val;
                }
            }
        }
        output[i * output_rows * output_size] = max_val;
    }
    // cout << output[a * output_rows * layer_size] << "\t\t" << a << endl;
}

void print_output(float *buffer, int rows, int cols, int depth = 1, int number = 1)
{
    for (int a = 0; a < number; a++)
    {
        // For each row, we print all depth layers side by side
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < depth; k++) // Loop over each layer
            {
                for (int j = 0; j < cols; j++) // Print one row of current layer
                {
                    cout << buffer[(a * depth * rows * cols) + (k * rows * cols) + (i * cols) + j] << "\t";
                    // cout<<((a * depth * rows * cols) + (k * rows * cols) + (i * cols) + j);
                }
                cout << "|"; // Add space between layers
            }
            cout << endl; // Move to the next row
        }
        cout << "==========================" << endl; // Separator between different outputs (if 'number' > 1)
    }
}

void shift_buffer(float *buffer, short kernel_size, short input_channels, short input_size)
{
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            for (int k = 0; k < input_channels; k++)
            {
                if (j < kernel_size - 1)
                    buffer[(i * kernel_size) + (j) + (k * input_size * kernel_size)] =
                        buffer[(i * kernel_size) + ((j + 1)) + (k * input_size * kernel_size)];
                else
                    buffer[(i * kernel_size) + (j) + (k * input_size * kernel_size)] = 0.0f;
            }
        }
    }
}
void print_model(Model Layer)
{
    cout << " rows: " << (int)Layer.rows << endl
         << " cols: " << (int)Layer.cols << endl
         << " channels: " << (int)Layer.channels << endl
         << " padding: " << (int)Layer.padding << endl
         << " stride: " << (int)Layer.stride << endl
         << " size: " << (int)Layer.size << endl;
}
void XNORCONV(const float *input, short input_channels, short input_size,
              const uint8_t *filter, short filter_rows, short no_filters,
              float *output, short layer_size, short output_rows, const float *alpha, const float *mean, const float *variance, const float *gamma, const float *beta)
{
    int n = filter_rows * filter_rows * input_channels; // Total elements in each filter
    for (short a = 0; a < no_filters; a++)
    {
        float sum = 0.0f;
        float K = 0.0f;
        // Apply the filter to the input (convolution window)
        for (short i = 0; i < input_channels; i++)
        {
            for (short k = 0; k < filter_rows; k++)
            {
                for (short l = 0; l < filter_rows; l++)
                {
                    // Get input value and binarize it
                    int input_index = i * filter_rows * input_size + k * filter_rows + l;
                    // Applying batch normalization
                    float input_value = gamma[i] * ((input[input_index] - mean[i]) / sqrt(variance[i] + 1e-4f)) + beta[i];
                    input_value = (input_value > 0) ? input_value : 0;
                    bool temp_input = (input_value > 0) ? 1 : 0;

                    // Compute filter index
                    int filter_index = a * input_channels * filter_rows * filter_rows + i * filter_rows * filter_rows + k * filter_rows + l;

                    // Access the correct bit in the binary filter
                    bool temp_filter = (filter[filter_index / 8] & (1 << (filter_index % 8))) ? 1 : 0;

                    // XNOR operation (equivalent to checking equality between binary values)
                    sum += !(temp_input ^ temp_filter);

                    // Accumulate absolute value of the input for K normalization
                    K += (input_value);
                }
            }
        }
        output[a * output_rows * layer_size] = ((2 * sum - n) * K * alpha[a]) / (n);
    }
}
void Conv2(float *input, short channels, short input_size, const float *filter, short filter_rows, short no_filters, float *output, short layer_size, short output_rows)
{
    for (short a = 0; a < no_filters; a++)
    {
        float sum = 0.0f;

        for (short i = 0; i < channels; i++)
        {
            for (short k = 0; k < filter_rows; k++)
            {
                for (short l = 0; l < filter_rows; l++)
                {
                    float elm1 = input[i * filter_rows * input_size + k * filter_rows + l];
                    elm1 = (elm1 > 0) ? elm1 : 0;
                    float elm2 = filter[a * channels * filter_rows * filter_rows + i * filter_rows * filter_rows + k * filter_rows + l];
                    sum += elm1 * elm2;
                    // cout<<elm1<<" "<<elm2<<endl;
                }
            }
        }
        

        output[a * output_rows * layer_size] = sum;
    }
}

