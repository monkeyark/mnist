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


string get_str_between_two_str(const string &s, const string &start_delim, const string &stop_delim)
{
	unsigned first_delim_pos = s.find_first_of(start_delim);
	unsigned end_pos_of_first_delim = first_delim_pos + start_delim.length();
	unsigned last_delim_pos = s.find_first_of(stop_delim, end_pos_of_first_delim);
	
	return s.substr(end_pos_of_first_delim, last_delim_pos - end_pos_of_first_delim);
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
	string path = "trained_weight1.txt";
	string text = read_file(path);
	string subtext = text;

	text = "NEW";

	// vector<string> modle;

	cout << text << endl;
	cout << subtext << endl;

	// read_model(text);
	
	// cout << text << endl;

	// string s = "aaa:bbbbcc;ceee";
	// string ss = get_str_between_two_str(s, ":", ";");

	// cout << "s:  " << s << endl;
	// cout << "ss: " << ss << endl;

	return 0;
}