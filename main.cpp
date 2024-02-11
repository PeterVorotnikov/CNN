#include <iostream>
#include <string>
#include <algorithm>
#include "CNN.h"

using namespace std;

int main() {
	vector<string> labels = {
		"T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
	};

	CNN cnn;
	cnn.load("model");

	ifstream file("fashion-mnist_test.csv");
	string density = "¶@ØÆMåBNÊßÔR#8Q&mÃ0À$GXZA5ñk2S%±3Fz¢yÝCJf1t7ªLc¿+?(r/¤²!*;\"^:,'.`";
	reverse(density.begin(), density.end());
	int r = 54;
	for (int i = 0; i < r; i++) {
		string s;
		getline(file, s);
	}
	int label;
	file >> label;
	cout << "Class: " << labels[label] << "\n\n";
	vector<vector<vector<double>>> image(1, vector<vector<double>>(28, vector<double>(28)));
	for (int r = 0; r < 28; r++) {
		for (int c = 0; c < 28; c++) {
			int val;
			file >> val;
			image[0][r][c] = (double)val / 255.0;
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
	cout << "\n";
	vector<double> out = cnn.predict(image);
	for (int i = 0; i < out.size(); i++) {
		cout << labels[i] << ": " << out[i] << "\n";
	}
	cout << "\n\n\n";
	file.close();
}