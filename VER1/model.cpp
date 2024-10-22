#include <iostream>
#include "help.cpp"
#include "weights.hpp"
#include "bn_variables.hpp"
#include "alphas.hpp"
#include "binary_weights.hpp"
 
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
 
#define INPUT_SIZE 224
#define INPUT_CHANNELS 3
void normalize(cv::Mat &image, const float mean[3], const float std[3]) {
    // Ensure the image is in floating point format
    image.convertTo(image, CV_32F, 1.0 / 255);  // Convert to [0, 1] range if the image is in [0, 255]

    // Split the image into 3 channels (B, G, R)
    cv::Mat channels[3];
    cv::split(image, channels);

    // Normalize each channel: (channel - mean) / std
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    // Merge the channels back
    cv::merge(channels, 3, image);
}
int main(int argc, char** argv)
{
    Model Layer[14];
    Layer[0] = Model{INPUT_SIZE, 11, 3, 2, 4};
    Layer[1] = Model{55, 3, 96, 0, 2};
    Layer[2] = Model{27, 5, 96, 2, 1};
    Layer[3] = Model{27, 3, 256, 0, 2};
    Layer[4] = Model{13, 3, 256, 1, 1};
    Layer[5] = Model{13, 3, 384, 1, 1};
    Layer[6] = Model{13, 3, 384, 1, 1};
    Layer[7] = Model{13, 3, 256, 0, 1};
    Layer[8] = Model{6, 6, 256, 0, 2};
    Layer[9] = Model{1, 1, 4096, 0, 1};
    Layer[10] = Model{1, 1, 4096, 0, 1};
    // print_model(Layer[3]);
    // Initialize the input array with values
    // short INPUT_SIZE = 5, INPUT_CHANNELS = 3;
    short kernel_size = 11, output_channels = 4;
    float input[INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS];
        // Load the image from the given path
    Mat image = imread(argv[1], IMREAD_COLOR);
    Mat resizedImage;
    resize(image, resizedImage, Size(224, 224));
    float mean[3] = {0.485, 0.456, 0.406};  // Mean for R, G, B
    float std[3] = {0.229, 0.224, 0.225};   // Standard deviation for R, G, B
    normalize(resizedImage, mean, std);
    memcpy(input, resizedImage.data, resizedImage.total() * resizedImage.elemSize());

    short padding0 = 2, stride0 = 4;
    short padding1 = 2, stride1 = 1;
    short padding2 = 0, stride2 = 1;
    short padding3 = 1, stride3 = 1;
    short padding4 = 1, stride4 = 1;
    // print_model(Layer[1]);
    // Fill the input array with values

    short layer0_size = ((INPUT_SIZE + 2 * Layer[0].padding));
    float layer0[layer0_size * kernel_size * INPUT_CHANNELS] = {0.0};

    float layer1[55 * 3 * 96] = {0.0};
    float layer2[Layer[2].size * Layer[2].cols * Layer[2].channels] = {0.0};
    float layer3[27 * 3 * 256] = {0.0};
    float layer4[15 * 3 * 256] = {0.0};
    float layer5[15 * 3 * 384] = {0.0};
    float layer6[15 * 3 * 384] = {0.0};
    float layer7[15 * 3 * 256] = {0.0};
    float layer8[6 * 6 * 256] = {0.0};
    float layer9[4096] = {0.0};
    float layer10[4096] = {0.0};
    float output[2] = {0.0};
    short rows = 0, cols = 0, rows2 = 0, cols2 = 0;
    short rows3 = 0, cols3 = 0, rows4 = 0, cols4 = 0;
    short rows5 = 0, cols5 = 0, rows6 = 0, cols6 = 0;
    short rows7 = 0, cols7 = 0, rows8 = 0, cols8 = 0;
    short rows9 = 0, cols9 = 0, rows10 = 0, cols10 = 0;
    // Perform convolution
    for (short j = 0; j < 323; j++)
    {
        // cout<<j;
        if ((j > Layer[0].cols - 1) && (j < layer0_size))
        {
            // print_output(layer0, layer0_size, kernel_size, output_channels, 1);
            shift_buffer(layer0, Layer[0].cols, INPUT_CHANNELS, layer0_size);
            // print_output(layer0, layer0_size, Layer[0].cols, output_channels, 1);
        }
        if (((rows / Layer[0].stride) > (Layer[1].cols - 2) - (2 * Layer[1].padding)) && ((rows / Layer[0].stride) < Layer[1].size - 2 * Layer[1].padding) && ((j + 1) % Layer[0].stride == 0))
        {

            // cout << rows << "\t" << (int)Layer[1].cols << endl;
            shift_buffer(layer1, Layer[1].cols, Layer[1].channels, Layer[1].size);
        }
        if (((rows2 / Layer[1].stride) > (Layer[2].cols + 1) - (2 * Layer[2].padding)) && ((rows2 / (Layer[0].stride)) < Layer[2].size - Layer[2].padding - 1) && ((j + 1) % (Layer[0].stride * Layer[1].stride) == 0))
        {
            // cout << rows2;
            shift_buffer(layer2, Layer[2].cols, Layer[2].channels, Layer[2].size);
        }
        if ((rows3 > (Layer[3].cols - 2)) && (rows3 < Layer[3].size) && ((j + 1) % (Layer[0].stride * Layer[1].stride) == 0))
        {
            shift_buffer(layer3, Layer[3].cols, Layer[3].channels, Layer[3].size);
        }
        if ((rows4 > (Layer[4].cols - 1) - (2 * Layer[4].padding)) && ((rows4 / Layer[3].stride) < Layer[4].rows) && (((j + 1) % 16) == 0))
        {
            shift_buffer(layer4, Layer[4].cols, Layer[4].channels, Layer[4].size);
        }
        if ((rows5 > (Layer[5].cols - 1) - (2 * Layer[5].padding)) && ((rows5) < Layer[5].rows) && (((j + 1) % 16) == 0))
        {
            shift_buffer(layer5, Layer[5].cols, Layer[5].channels, Layer[5].size);
        }
        if ((rows6 > (Layer[6].cols - 1) - (2 * Layer[6].padding)) && ((rows6) < Layer[6].rows) && (((j + 1) % 16) == 0))
        {
            shift_buffer(layer6, Layer[6].cols, Layer[6].channels, Layer[6].size);
        }
        if (rows7 > (Layer[7].cols - 2) && ((rows7) < Layer[7].rows - 1) && (((j + 1) % 16) == 0))
        {
            shift_buffer(layer7, Layer[7].cols, Layer[7].channels, Layer[7].size);
        }

        // Update buffer with the new column data

        for (int i = 0; i < Layer[0].size + Layer[1].padding + Layer[2].padding + Layer[3].padding + Layer[4].padding + 100; i++)
        {
            //---------------------LAYER0------------------------------//
            for (short k = 0; k < Layer[0].channels; k++)
            {
                if ((i < Layer[0].size) && (j < INPUT_SIZE + Layer[0].padding))
                {

                    if (i < Layer[0].padding || j < Layer[0].padding || i > INPUT_SIZE + Layer[0].padding - 1 || j > INPUT_SIZE + Layer[0].padding - 1)
                    {
                        if (j < Layer[0].cols)
                            layer0[(i * Layer[0].cols) + (j) + (k * layer0_size * Layer[0].cols)] = 0;
                        else
                            layer0[(i * Layer[0].cols) + (Layer[0].cols - 1) + (k * layer0_size * Layer[0].cols)] = 0;
                    }
                    else
                    {
                        short row_index = i - Layer[0].padding;
                        short col_index = j - Layer[0].padding;

                        if (j < Layer[0].cols)
                            layer0[(i * Layer[0].cols) + (j) + (k * layer0_size * Layer[0].cols)] = (input[(row_index * INPUT_SIZE) + col_index + (k * INPUT_SIZE * INPUT_SIZE)]);
                        else
                            layer0[(i * Layer[0].cols) + (Layer[0].cols - 1) + (k * layer0_size * Layer[0].cols)] = (input[(row_index * INPUT_SIZE) + col_index + (k * INPUT_SIZE * INPUT_SIZE)]);
                    }
                }
            }
            //---------------------LAYER1------------------------------//
            if ((j >= Layer[0].cols - 1) && (i >= Layer[0].cols - 1))
            {
                rows = j - Layer[0].cols + 1;
                cols = i - Layer[0].cols + 1;
                if (((rows / Layer[0].stride) < Layer[1].size - 2 * Layer[1].padding) && ((cols / Layer[0].stride) < Layer[1].size - (2 * Layer[1].padding)) && ((rows % Layer[0].stride == 0) && (cols % Layer[0].stride == 0)))
                {
                    if ((rows / Layer[0].stride) + Layer[1].padding < Layer[1].cols)
                    {
                        Convolution(&layer0[cols * Layer[0].cols], 3, 228, conv0, 11, 96, &layer1[((cols / Layer[0].stride) + Layer[1].padding) * Layer[1].cols + ((rows / Layer[0].stride) + Layer[1].padding)], Layer[1].size, Layer[1].cols, Batch_0_running_mean, Batch_0_running_var, Batch_0_weight, Batch_0_bias);
                    }
                    else
                    {
                        Convolution(&layer0[cols * Layer[0].cols], 3, 228, conv0, 11, 96, &layer1[((cols / Layer[0].stride) + Layer[1].padding) * Layer[1].cols + (Layer[1].cols - 1)], Layer[1].size, Layer[1].cols, Batch_0_running_mean, Batch_0_running_var, Batch_0_weight, Batch_0_bias);
                    }
                }
                //---------------------LAYER2------------------------------//

                if ((((rows / Layer[0].stride) + Layer[1].padding) > Layer[1].cols - 2) &&
                    (((cols / Layer[0].stride) + Layer[1].padding) > Layer[1].cols - 2) &&
                    ((rows % (Layer[0].stride * Layer[1].stride) == 0) && (cols % (Layer[0].stride * Layer[1].stride) == 0)))
                {
                    rows2 = ((rows) / Layer[1].stride) - Layer[1].cols + Layer[1].padding - 1;
                    cols2 = ((cols) / Layer[1].stride) - Layer[1].cols + Layer[1].padding - 1;
                    // print_output(layer2, layer2_size, kernel_size, output_channels, 1);
                    // cout << j;
                    // print_output(layer1, 55, 3);
                    if (((rows2 / (Layer[0].stride)) < Layer[2].size - 2 * Layer[2].padding) && ((cols2 / Layer[0].stride) < Layer[2].size - (2 * Layer[2].padding)))
                    {
                        if (rows2 / Layer[1].stride + Layer[2].padding < Layer[2].cols)
                        { //------------  i  -----------------------

                            MaxPooling(&layer1[(cols2 / 2) * Layer[1].cols], Layer[1].channels, Layer[1].size, Layer[1].cols, &layer2[((cols2 / (Layer[0].stride)) + Layer[2].padding) * Layer[2].cols + ((rows2 / (Layer[0].stride)) + Layer[2].padding)], Layer[2].size, Layer[2].cols);
                        }
                        else
                            MaxPooling(&layer1[(cols2 / 2) * Layer[1].cols], Layer[1].channels, Layer[1].size, Layer[1].cols, &layer2[((cols2 / Layer[0].stride) + Layer[2].padding) * Layer[2].cols + (Layer[2].cols - 1)], Layer[2].size, Layer[2].cols);
                    }

                    //     //---------------------LAYER3------------------------------//

                    if ((((rows2 / Layer[0].stride) + Layer[2].padding) >= Layer[2].cols - 1) && (((cols2 / Layer[0].stride) + Layer[2].padding) >= Layer[2].cols - 1))
                    {
                        rows3 = (rows2 - Layer[2].cols - Layer[2].padding - 1) / 4;
                        cols3 = (cols2 - Layer[2].cols - Layer[2].padding - 1) / 4;
                        // print_output(layer2, 31, 5, 1, 1);
                        if ((rows3 < Layer[3].size) && (cols3 < Layer[3].size))
                        {
                            // cout << rows3 << "\t" << cols3 * Layer[2].cols << endl;
                            if (rows3 < 3)
                                XNORCONV(&layer2[cols3 * Layer[2].cols], 96, 31, conv2, 5, 256, &layer3[cols3 * Layer[3].cols + rows3], Layer[3].size, Layer[3].cols, alpha0, Batch_2_running_mean, Batch_2_running_var, Batch_2_weight, Batch_2_bias);
                            else
                                XNORCONV(&layer2[cols3 * Layer[2].cols], 96, 31, conv2, 5, 256, &layer3[(cols3 * Layer[3].cols) + (Layer[3].cols - 1)], Layer[3].size, Layer[3].cols, alpha0, Batch_2_running_mean, Batch_2_running_var, Batch_2_weight, Batch_2_bias);
                        }
                        //---------------------LAYER4------------------------------//
                        if ((rows3 >= Layer[3].cols - 1) && (cols3 >= Layer[3].cols - 1) && ((rows3 % (Layer[3].stride) == 0) && (cols3 % (Layer[3].stride) == 0)))
                        {
                            rows4 = (rows3 - Layer[3].cols - Layer[3].padding + 1);
                            cols4 = (cols3 - Layer[3].cols - Layer[3].padding + 1);
                            if (((rows4 / (Layer[3].stride)) < Layer[4].size - 2 * Layer[4].padding) && ((cols4 / Layer[3].stride) < Layer[4].size - (2 * Layer[4].padding)))
                            {

                                if (rows4 + Layer[4].padding < Layer[4].cols)
                                    MaxPooling(&layer3[cols4 * Layer[3].cols], Layer[3].channels, Layer[3].size, Layer[3].cols, &layer4[((cols4 / (Layer[3].stride)) + Layer[4].padding) * Layer[3].cols + ((rows4 / (Layer[3].stride)) + Layer[4].padding)], Layer[4].size, Layer[4].cols);
                                else
                                    MaxPooling(&layer3[cols4 * Layer[3].cols], Layer[3].channels, Layer[3].size, Layer[3].cols, &layer4[((cols4 / (Layer[3].stride)) + Layer[4].padding) * Layer[3].cols + Layer[4].cols - 1], Layer[4].size, Layer[4].cols);
                            }

                            //---------------------LAYER5------------------------------//
                            if (((rows4 + Layer[4].padding) >= Layer[4].cols - 1) && ((cols4 + Layer[4].padding) >= Layer[4].cols - 1))
                            {
                                rows5 = (rows4 - Layer[4].cols + 1) / 2;
                                cols5 = (cols4 - Layer[4].cols + 1) / 2;
                                if ((rows5 < Layer[5].size - 2 * Layer[5].padding) && (cols5 < Layer[5].size - (2 * Layer[5].padding)))
                                {
                                    if (rows5 + Layer[5].padding < Layer[5].cols)
                                        XNORCONV(&layer4[cols5 * Layer[4].cols], 256, 15, conv3, 3, 384, &layer5[(cols5 + Layer[4].padding) * Layer[5].cols + (rows5 + Layer[4].padding)], Layer[5].size, Layer[5].cols, alpha1, Batch_4_running_mean, Batch_4_running_var, Batch_4_weight, Batch_4_bias);
                                    else
                                        XNORCONV(&layer4[cols5 * Layer[4].cols], 256, 15, conv3, 3, 384, &layer5[(cols5 + Layer[4].padding) * Layer[5].cols + (Layer[5].cols - 1)], Layer[5].size, Layer[5].cols, alpha1, Batch_4_running_mean, Batch_4_running_var, Batch_4_weight, Batch_4_bias);
                                }
                                // print_output(layer5, 15, 3);
                                if (((rows5 + Layer[5].padding) >= Layer[5].cols - 1) && ((cols5 + Layer[5].padding) >= Layer[5].cols - 1))
                                {
                                    rows6 = (rows5 - Layer[5].cols + 1 + Layer[5].padding);
                                    cols6 = (cols5 - Layer[5].cols + 1 + Layer[5].padding);
                                    if ((rows6 < Layer[6].rows) && (cols6 < Layer[6].rows))
                                    {
                                        if (rows6 + Layer[6].padding < Layer[6].cols)
                                            XNORCONV(&layer5[cols6 * Layer[4].cols], 384, 15, conv4, 3, 384, &layer6[(cols6 + Layer[5].padding) * Layer[6].cols + (rows6 + Layer[5].padding)], Layer[6].size, Layer[6].cols, alpha2, Batch_5_running_mean, Batch_5_running_var, Batch_5_weight, Batch_5_bias);
                                        else
                                            XNORCONV(&layer5[cols6 * Layer[4].cols], 384, 15, conv4, 3, 384, &layer6[(cols6 + Layer[5].padding) * Layer[6].cols + (Layer[6].cols - 1)], Layer[6].size, Layer[6].cols, alpha2, Batch_5_running_mean, Batch_5_running_var, Batch_5_weight, Batch_5_bias);
                                        // cout << rows6 << "\t" << cols6 << endl;
                                        // print_output(layer6, 15, 3);
                                    }
                                    if (((rows6 + Layer[6].padding) >= Layer[6].cols - 1) && ((cols6 + Layer[6].padding) >= Layer[6].cols - 1))
                                    {
                                        rows7 = (rows6 - Layer[6].cols + 1 + Layer[6].padding);
                                        cols7 = (cols6 - Layer[6].cols + 1 + Layer[6].padding);
                                        if ((rows7 < Layer[7].rows) && (cols7 < Layer[7].rows))
                                        {
                                            if (rows7 + Layer[7].padding < Layer[7].cols)
                                                XNORCONV(&layer6[cols7 * Layer[6].cols], 384, 15, conv5, 3, 256, &layer7[(cols7 + Layer[7].padding) * Layer[7].cols + (rows7 + Layer[7].padding)], Layer[7].size, Layer[7].cols, alpha3, Batch_6_running_mean, Batch_6_running_var, Batch_6_weight, Batch_6_bias);
                                            else
                                                XNORCONV(&layer6[cols7 * Layer[6].cols], 384, 15, conv5, 3, 256, &layer7[(cols7 + Layer[7].padding) * Layer[7].cols + (Layer[7].cols - 1)], Layer[7].size, Layer[7].cols, alpha3, Batch_6_running_mean, Batch_6_running_var, Batch_6_weight, Batch_6_bias);
                                            // cout << rows7 << "\t" << cols7 << endl;
                                        }
                                        if (((rows7 + Layer[7].padding) >= Layer[7].cols - 1) && ((cols7 + Layer[7].padding) >= Layer[7].cols - 1) && ((rows7 % (Layer[8].stride) == 0) && (cols7 % (Layer[8].stride) == 0)))
                                        {

                                            rows8 = rows7 - Layer[7].cols + 1;
                                            cols8 = cols7 - Layer[7].cols + 1;
                                            if ((rows8 / Layer[8].stride) < Layer[8].cols)
                                                MaxPooling(&layer7[cols8 * Layer[7].cols], Layer[7].channels, Layer[7].size, Layer[7].cols, &layer8[((cols8 / (Layer[8].stride)) + Layer[8].padding) * Layer[8].cols + ((rows8 / (Layer[8].stride)) + Layer[8].padding)], Layer[8].size, Layer[8].cols);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if ((rows8 / Layer[8].stride) == Layer[8].cols)
        {
            // rows9 = rows8 - Layer[8].cols + 1;
            // cols9 = cols8 - Layer[8].cols + 1;
            XNORCONV(layer8, 256, 6, conv6, 6, 4096, layer9, 1, 1, alpha4, Batch_8_running_mean, Batch_8_running_var, Batch_8_weight, Batch_8_bias);
            XNORCONV(layer9, 4096, 1, conv7, 1, 4096, layer10, 1, 1, alpha5, Batch_9_running_mean, Batch_9_running_var, Batch_9_weight, Batch_9_bias);
            Conv2(layer10, 4096, 1, conv8, 1, 2, output, 1, 1);
            // print_output(layer8, 6, 6);
            cout<<output[0]<<endl;
            cout<<output[1]<<endl;
            cout<<j<<endl;
            // cout<<i<<endl;
        }
    }
    return 0;
}
