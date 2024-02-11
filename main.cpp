#include <iostream>
#include <string>
#include <algorithm>
#include "CNN.h"

using namespace std;

int main() {
	int nOfSamples = 60000;

	int setRand = 0;
	cout << "Set rand:\n";
	cin >> setRand;
	srand(setRand);

	string dataFileName = "fashion-mnist_train.csv";

	string modelName;
	cout << "Enter the file name of the loading model:\n";
	cin >> modelName;

	int epochs;
	cout << "Enter the number of epochs:\n";
	cin >> epochs;

	CNN cnn;
	if (modelName != "-") {
		cnn.load(modelName);
	}

	ofstream outFile1("out1.csv");
	outFile1 << "epoch,sample,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,label,loss\n";
	outFile1.close();

	ofstream outFile2("out2.csv");
	outFile2 << "epoch,loss,accuracy\n";
	outFile2.close();

	for (int e = 1; e <= epochs; e++) {

		ifstream dataFile(dataFileName);
		string s;
		getline(dataFile, s);

		double sumLoss = 0;
		int nOfCorrectAnswers = 0;

		for (int k = 1; k <= nOfSamples; k++) {

			int label;
			dataFile >> label;
			vector<vector<vector<double>>> image(1, vector<vector<double>>(28, vector<double>(28)));
			for (int r = 0; r < 28; r++) {
				for (int c = 0; c < 28; c++) {
					int value;
					dataFile >> value;
					image[0][r][c] = (double)value / 255.0;
				}
			}

			cnn.fit(image, label);
			double loss = cnn.getLoss(image, label);
			vector<double> out = cnn.predict(image);

			outFile1.open("out1.csv", ios_base::app);
			outFile1 << e << "," << k << ",";
			for (int i = 0; i < out.size(); i++) {
				outFile1 << out[i] << ",";
			}
			outFile1 << label << ",";
			outFile1 << loss << "\n";
			outFile1.close();

			sumLoss += loss;
			if (out[label] > 0.5) {
				nOfCorrectAnswers++;
			}

			cout << "Epoch: " << e << ", sample: " << k << ", loss = " << sumLoss / (double)k <<
				", accuracy = " << (double)nOfCorrectAnswers / (double)k << "\n";

			if (k % 100 == 0) {
				cout << cnn.save("models\\epoch" + to_string(e) + "samples" + to_string(k))<< "\n";
			}
		}

		outFile2.open("out2.csv", ios_base::app);
		outFile2 << e << "," << sumLoss / (double)nOfSamples << "," <<
			(double)nOfCorrectAnswers / (double)nOfSamples << "\n";
		outFile2.close();

		dataFile.close();

	}

	int x;
	cin >> x;

	return 0;
}