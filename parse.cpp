#include "parse.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <vector>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::regex;
using std::regex_match;
using std::regex_search;
using std::cmatch;
using std::smatch;


class CNN_LAYER
{
};

class CONV2D : public CNN_LAYER
{
	public:
		int depth;
		int filters;
		int kernel_size;
		int stride;
		vector<vector<vector<double> > > plain_kernals;
};

class LINEAR : public CNN_LAYER
{
	public:
		int in_feature;
		int out_feature;
		vector<vector<vector<double> > > plain_weights;
};

class CNN_MODEL
{
	public:
		string name;
		int num_conv2d;
		int num_linear;
		vector<CNN_LAYER> layer;
};

string get_str_between_two_str(const string s, const string start_delim, const string stop_delim)
{
	string substring;
	unsigned first_delim_pos = s.find(start_delim);
	unsigned end_pos_of_first_delim = first_delim_pos + start_delim.length();
	unsigned last_delim_pos = s.find(stop_delim);
	unsigned end_pos_of_last_delim = last_delim_pos + stop_delim.length();

	if (end_pos_of_first_delim >= s.length() || end_pos_of_last_delim >= s.length())
		substring = "";
	else
		substring = s.substr(end_pos_of_first_delim, last_delim_pos - end_pos_of_first_delim);
	
	return substring;
}

string read_file(string file_path)
{
	ifstream file(file_path);
	string text, line;
	while (getline(file, line))
	{
		text = text + line + "\n";
	}
	file.close();
	return text;
}

vector<CNN_MODEL> read_model(string file_text)
{
	string text = file_text;
	string delim_start = "MODEL_START";
	string delim_end = "MODEL_END";
	string substring = text;

	vector<string> model_string;
	while (text.length() != 0)
	{
		substring = get_str_between_two_str(text, delim_start, delim_end);
		if (substring == "")
			break;
		model_string.push_back(substring);
		text = text.substr(text.find(delim_end) + delim_end.length());
	}

	vector<CNN_MODEL> cnn_model;
	for (auto m: model_string)
	{
		// cout << m << endl;
		// cout << "===========================================================================" << endl;
		// read_model(m);
		smatch sm1, sm2;
		regex cnn ("(CNN)(.*)(_)(.*)");
		regex_search (m, sm1, cnn);
		regex tensor ("(tensor([)(.*)(],\ndtype=torch.float64)");
		regex_search (m, sm2, tensor);
		cout << sm2[0] << endl;

		CNN_MODEL model;
		model.num_conv2d = stoi(sm1.str(2));
		model.num_linear = stoi(sm1.str(4));


		cnn_model.push_back(model);
	}

	return cnn_model;
}


int main()
{
	string path = "trained_weight3.txt";
	string file_text = read_file(path);

	vector<CNN_MODEL> model = read_model(file_text);


	return 0;
}