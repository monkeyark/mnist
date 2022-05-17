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
using std::smatch;


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
	
	// cout << s << first_delim_pos << "    " << last_delim_pos << endl;
	// cout << substring << endl;
	// cout << "----------------------------------------------------------------------" << endl;
	return substring;
}

void read_model(string s)
{
	smatch sm;
	regex e ("(CNN)(.*)");
	regex_match (s, sm, e);
	for (unsigned i=0; i<sm.size(); ++i)
	{
		cout << "i:" << i << " ";
		cout << "[" << sm[i] << "] ";
	}
	// if (regex_match (s, e))
	// 	cout << regex_match ("subject", regex("(sub)(.*)"));
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


int main()
{
	string path = "trained_weight2.txt";
	string file_text = read_file(path);
	string text = file_text;
	string delim_start = "MODEL_START";
	string delim_end = "MODEL_END";

	// printf("%d==\n", text[text.length()-2]);
	// text = "fdsafMODEL_START\n a\nMODEL_END\nMODEL_START\n bb\nMODEL_END\nMODEL_START\n ccc\nMODEL_END\nssss\n";
	// text = "0123456789\n";
	string substring = text;
	vector<string> model;
	while (text.length() != 0)
	{
		substring = get_str_between_two_str(text, delim_start, delim_end);
		if (substring == "")
			break;

		model.push_back(substring);
		text = text.substr(text.find(delim_end) + delim_end.length());
	}

	for (auto m: model)
	{
		cout << m << endl;
		cout << "===========================================================================" << endl;

	}

	return 0;
}