#include <iostream>
#include <vector>

using namespace std;

typedef struct
{
    int convl_layers;
    int fully_layers;
    int input_side;
    int input_channel;
    size_t slot_count;
    int images_parallel_number; // the number of parallel images in one forward pass
    int combined_input_side;
    int combined_matrix_dimension;
    int combined_stride;

    // how many (d*d*images_parallel_number) can be packed in one ciphertext
    // d is the matrix side of the last convolutional layer
    int pack_parallel_number; 
    // combined_matrix_dimension*images_parallel_number
    int repeat_count;
    // the pack ratio: combined_matrix_dimension*pack_parallel_number
    int pack_ratio;

    vector<int> kernal_sides;
    vector<int> kernal_strides;
    vector<int> kernal_numbers;
    vector<int> kernal_channels;
    vector<int> effective_kernal_sides;

} _parameters;

_parameters para;

// The randomly generated values are in (0, 1)
void generateMatrix(std::vector<std::vector<std::vector<double> > > &values, int channel, int side){
    values = std::vector<std::vector<std::vector<double> > >(channel,
                        std::vector<std::vector<double> >(side, std::vector<double>(side)));
    for(int i = 0; i < channel; i++){
        for(int j = 0; j < side; j++){
            for(int k = 0; k < side; k++){
                values[i][j][k] = (double)rand() / RAND_MAX;
            }
        }
    }
}

void generateMultiMatrix(std::vector<std::vector<std::vector<std::vector<double> > > > &values, int parallel_number, int channel, int side){
    // values = std::vector<std::vector<std::vector<std::vector<double>>>>(parallel_number, std::vector<std::vector<std::vector<double>>>(channel,
                        // std::vector<std::vector<double>>(side, std::vector<double>(side))));
    for(int p = 0; p < parallel_number; p++){
        for(int i = 0; i < channel; i++){
            for(int j = 0; j < side; j++){
                for(int k = 0; k < side; k++){
                    values[p][i][j][k] = (double)rand() / RAND_MAX;
                }
            }
        }
    }
    
}

void generateKernalInOneDimension(std::vector<double> &kernal, int channel, int side){
    kernal = std::vector<double>(channel*side*side);
    for(int i = 0; i < channel*side*side; i++){
        kernal[i] = (double)rand()/RAND_MAX;
    }
}

void generateMultiPlainKernals(std::vector<std::vector<double> > &kernal, int kernal_number, int channel, int side){
    for(int num = 0; num < kernal_number; num++){
        for(int i = 0; i < channel*side*side; i++){
            kernal[num][i] = (double)rand()/RAND_MAX;
        }
    }
}

void initModelKernalsAndWeights(vector<int> kernal_numbers, vector<int> kernal_channels, vector<int> kernal_sides, vector<int> fully_layer_neurons,
								vector<vector<vector<double> > > &plain_kernals, vector<vector<vector<double> > > &plain_weights)
{
	// Generate filters
	for (int i = 0; i < para.convl_layers; i++)
	{
		plain_kernals[i] = vector<vector<double> >(kernal_numbers[i], vector<double>(kernal_channels[i] * kernal_sides[i] * kernal_sides[i]));
		generateMultiPlainKernals(plain_kernals[i], kernal_numbers[i], kernal_channels[i], kernal_sides[i]);
	}
	// Generate fully connecte layer weights
	for (int layer = 0; layer < para.fully_layers; layer++)
	{
		int in_dimension = fully_layer_neurons[layer];
		int out_dimension = fully_layer_neurons[layer + 1];
		plain_weights[layer] = vector<vector<double> >(in_dimension, vector<double>(out_dimension));
		generateMultiPlainKernals(plain_weights[layer], in_dimension, out_dimension, 1);
	}
}

int main()
{
	// convolutional output dimensions, which is the input dimension of the first fully connected layer weights
    int convol_output_dimension = para.combined_matrix_dimension * para.kernal_numbers[para.convl_layers - 1];
    
    // 2 fully connected layers
    para.fully_layers = 2;
    // the hidden layer has 64 neurons, output layer has 10 neurons
    vector<int> fully_layer_neurons{convol_output_dimension, 64, 10};

    // Generate kernals and weights
    vector<vector<vector<double> > > plain_kernals(para.convl_layers);
    vector<vector<vector<double> > > plain_weights(para.fully_layers);
    initModelKernalsAndWeights(para.kernal_numbers, para.kernal_channels, para.kernal_sides, fully_layer_neurons, plain_kernals, plain_weights);

	return 0;
}
