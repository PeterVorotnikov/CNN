#include <iostream>
#include <string>
#include <algorithm>
#include "CNN2.h"

using namespace std;

int main() {
	cout << "Enter test data file name:\n";
	string fileName;
	cin >> fileName;
	fileName = "fashion-mnist_test.csv";

	cout << "Enter n of train samples:\n";
	int nOfTrainSamples;
	cin >> nOfTrainSamples;

	cout << "Enter n of test samples:\n";
	int nOfTestSamples;
	cin >> nOfTestSamples;

	cout << "Enter step:\n";
	int step;
	cin >> step;


	ofstream fileOut("output.csv");
	fileOut << "epoch,train_samples,loss,accuracy\n";
	fileOut.close();

	for (int epoch = 1; epoch <= 1; epoch++) {
		int trainSample = 0;
		if (epoch > 1) {
			trainSample = step;
		}
		while (trainSample <= 1) {

			CNN2 cnn;
			cnn.load("model");

			ifstream file(fileName);
			if (!file.is_open()) {
				return -1;
			}

			string s;
			getline(file, s);


			double loss = 0;
			double nOfCorrectAnswers = 0;

			for (int testSample = 1; testSample <= nOfTestSamples; testSample++) {

				int label;
				file >> label;

				vector<vector<vector<double>>> image(1, vector<vector<double>>(28, vector<double>(28)));
				for (int r = 0; r < 28; r++) {
					for (int c = 0; c < 28; c++) {
						int value;
						file >> value;
						image[0][r][c] = (double)value / 255.0;
					}
				}

				double currLoss = cnn.getLoss(image, label);
				loss += currLoss;
				if (cnn.predict(image)[label] > 0.5) {
					nOfCorrectAnswers++;
				}

				cout << "e = " << epoch << ", s = " << trainSample
					<< ", test sample = " << testSample << ", loss = "
					<< loss / (double)testSample << ", accuracy = "
					<< (double)nOfCorrectAnswers / (double)testSample << "\n";
			}

			fileOut.open("output.csv", ios_base::app);
			fileOut << epoch << "," << trainSample << "," << loss / (double)nOfTestSamples
				<< "," << (double)nOfCorrectAnswers / (double)nOfTestSamples << "\n";
			fileOut.close();


			file.close();
			trainSample += step;
		}
	}

	return 0;
}