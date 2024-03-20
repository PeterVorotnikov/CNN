#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include "CNN2.h"

using namespace std;

int main() {
	cout.precision(20);
	ifstream file("fashion-mnist_train.csv");
	string s;
	for (int i = 0; i < 155; i++) {
		getline(file, s);
	}
	vector<vector<vector<double>>> image(1, vector<vector<double>>(28, vector<double>(28)));
	int label;
	file >> label;
	for (int r = 0; r < 28; r++) {
		for (int c = 0; c < 28; c++) {
			int val;
			file >> val;
			image[0][r][c] = (double)val / 255.0;
		}
	}
	file.close();

	CNN2 cnn;
	cnn.load("epoch1samples0");
	cnn.predict(image);
	int i = 49, j = 6, k = 1;
	double h = 0.05;
	for (int i = 0; i < 780; i+=1) {
		for (int j = 0; j < 60; j+=1) {
			double f = cnn.getLoss(image, label);
			cnn.fullWeights[k][i][j] += h;
			//cnn.fullBiases[k][j] += h;
			double F = cnn.getLoss(image, label);
			cnn.fullWeights[k][i][j] -= 2 * h;
			//cnn.fullBiases[k][j] -= 2 * h;;
			double f_ = cnn.getLoss(image, label);
			cnn.fullWeights[k][i][j] += h;
			//cnn.fullBiases[k][j] += h;
			cnn.predict(image);
			cnn.backPropagation();
			cnn.backPropagation2();
			cout << i << " " << j << " " << cnn.fullWeightsDiff[k][i][j] / cnn.fullWeightsDiff2[k][i][j]
				<< " " << (F - f) / h / ((F - 2 * f + f_) / h / h) << endl;
			/*cout << i << " " << j << " " << cnn.fullBiasesDiff2[k][j]
				<< " " << ((F - 2 * f + f_) / h / h) << endl;*/
		}
	}
}