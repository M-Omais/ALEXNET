#include <iostream>
using namespace std;

#define input_size 6
#define kernel_size 3

void shift_buffer(float *buffer)
{
    // Shift all elements to the left (remove first column and shift remaining columns)
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < kernel_size - 1; j++)
        {
            buffer[i * kernel_size + j] = buffer[i * kernel_size + (j + 1)];
        }
    }
}

void print_output(float *buffer)
{
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            cout << buffer[i * kernel_size + j] << " ";
        }
        cout << endl;
    }
    cout << "---------------" << endl;
}



int main()
{
    int input[input_size][input_size];
    float buffer[input_size * kernel_size];
    unsigned int sum = 1;

    // Initialize input array
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            input[i][j] = sum++;
        }
    }

    // Perform sliding window operation

    for (int j = 0; j <= input_size - kernel_size; j++)
    {
        // Fill the buffer with the current window
        for (int k = 0; k < input_size; k++)
        {
            for (int l = 0; l < kernel_size; l++)
            {
                buffer[k * kernel_size + l] = static_cast<float>(input[k][j + l]);
            }
        }

        // Print the current window
        print_output(buffer);

        // Shift the buffer for the next column
        if (j < input_size - kernel_size)
        {
            shift_buffer(buffer);
        }
    }

    return 0;
}
