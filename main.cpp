#include <iostream>
#include <string>
#include <algorithm>
#include "CNN2.h"

using namespace std;

int main() {
	ifstream file("fashion-mnist_train.csv");
	string s;
	getline(file, s);
	vector<vector<vector<double>>> image(1, vector<vector<double>>(28, vector<double>(28)));
	for (int r = 0; r < 28; r++) {
		for (int c = 0; c < 28; c++) {
			int val;
			file >> val;
			image[0][r][c] = (double)val / 255.0;
		}
	}
	file.close();

	CNN2 cnn;
	cnn.predict(image);
	cnn.backPropagation();
	cnn.backPropagation2();
	int i = 25, j = 5;
	double h = 0.1;
	double f = cnn.getLoss(image, 0);
	cnn.outputWeights[i][j] += h;
	double fhplus = cnn.getLoss(image, 0);
	cnn.outputWeights[i][j] -= 2 * h;
	double fhminus = cnn.getLoss(image, 0);
	cnn.outputWeights[i][j] += h;
	cout << cnn.outputWeightsDiff2[i][j] << "\t" << (fhplus - 2 * f + fhminus) / h / h;
}