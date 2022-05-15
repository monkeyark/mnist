#include "parse.h"

#include <iostream>
#include <fstream>
#include <regex>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::regex;
using std::regex_match;
using std::smatch;

void read_model(string s)
{
	smatch sm;
	regex e ("(CNN)(.*)");
	cout << regex_match (s, sm, e);
	for (unsigned i=0; i<sm.size(); ++i)
	{
		std::cout << "[" << sm[i] << "] ";
	}
	// if (regex_match (s, e))
	// 	cout << regex_match ("subject", regex("(sub)(.*)"));

}

void read_file(string file_path)
{
	ifstream file(file_path);
	string text, line;
	while (getline(file, line))
	{
		text = text + line + "\n";
		read_model(line);
	}
	file.close();
	// cout << text << endl;
}


int main()
{
	string path = "trained_weight1.txt";
	read_file(path);
	cout << endl;
}