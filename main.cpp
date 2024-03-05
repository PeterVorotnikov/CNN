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
	int i = 50, j = 7;
	double h = 0.0001;
	double f = cnn.getLoss(image, 0);
	//cnn.outputWeights[i][j] += h;
	cnn.outputBiases[j] += h;
	double fhplus = cnn.getLoss(image, 0);
	//cnn.outputWeights[i][j] -= 2 * h;
	cnn.outputBiases[j] -= 2 * h;
	double fhminus = cnn.getLoss(image, 0);
	//cnn.outputWeights[i][j] += h;
	cnn.outputBiases[j] += h;
	cnn.backPropagation();
	cnn.backPropagation2();
	//cout << cnn.outputWeightsDiff2[i][j] << "\t" << (fhplus - 2 * f + fhminus) / h / h;
	cout << cnn.outputBiasesDiff2[j] << "\t" << (fhplus - 2 * f + fhminus) / h / h;
}