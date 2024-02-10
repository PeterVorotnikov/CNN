#include <iostream>
#include <string>
#include <algorithm>
#include "CNN.h"

using namespace std;

int main() {
	CNN cnn;

	/*for (int i = 0; i < 100; i++) {
		cnn.forwardPropagation();
		cnn.backPropagation();
		cout << cnn.convBiasesDiff[0][5] << " ";
		double loss1 = cnn.lossValue;
		cnn.convBiases[0][5] += 0.000001;
		cnn.forwardPropagation();
		double loss2 = cnn.lossValue;
		cout << (loss2 - loss1) / 0.000001 << endl;
		cout << loss1 << " " << loss2 << endl;
		cnn.convBiases[0][5] -= 0.000001;
		cnn.forwardPropagation();
		cnn.backPropagation();
		cnn.updateWeights();
	}*/

	/*for (int i = 0; i < 100; i++) {
		cnn.forwardPropagation();
		cnn.backPropagation();
		cnn.updateWeights();
		cnn.forwardPropagation();
		for (int j = 0; j < 10; j++) {
			cout << cnn.outputOutputs[j] << " ";
		}
		cout << "\t" << cnn.convBiases[0][0];
		cout << endl;
	}
	cnn.save("model.txt");*/

	/*cnn.load("model.txt");
	cnn.forwardPropagation();
	for (int i = 0; i < 10; i++) {
		cout << cnn.outputOutputs[i] << " ";
	}*/


	ifstream file("fashion-mnist_train.csv");
	string density = "¶@ØÆMåBNÊßÔR#8Q&mÃ0À$GXZA5ñk2S%±3Fz¢yÝCJf1t7ªLc¿+?(r/¤²!*;\"^:,'.`";
	reverse(density.begin(), density.end());
	int r = 160;
	for (int i = 0; i < r; i++) {
		string s;
		getline(file, s);
	}
	int label;
	file >> label;
	cout << "Label: " << label << "\n\n";
	for (int r = 0; r < 28; r++) {
		for (int c = 0; c < 28; c++) {
			int val;
			file >> val;
			int step = 255 / density.size();
			if (val > step * density.size()) {
				val = step * density.size();
			}
			int p = 0;
			int index = 0;
			for (int i = 0; i < density.size(); i++) {
				p += step;
				if (val <= p) {
					index = i;
					break;
				}
			}
			cout << density[index] << density[index];
		}
		cout << endl;
	}
	file.close();
}