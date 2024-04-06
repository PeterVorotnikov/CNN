#include "CNN.h"




//***************************************** Activation functions *******************


double CNN::convActivation(double x) {

	if (x >= 0) {

		return x;

	}

	else {

		return 0;

	}

}

double CNN::convActivationDiff(double x) {

	if (x >= 0) {

		return 1;

	}

	else {

		return 0;

	}

}

//double CNN::convActivation(double x) {
//
//	return 1.0 / (1.0 + exp(-x));
//
//}
//
//double CNN::convActivationDiff(double x) {
//
//	return convActivation(x) * (1 - convActivation(x));
//
//}











double CNN::layerActivation(double x) {

	return 1.0 / (1.0 + exp(-x));

}
double CNN::layerActivationDiff(double x) {

	return layerActivation(x) * (1 - layerActivation(x));

}











vector<double> CNN::softmaxOutputs(vector<double> softmaxInputs) {

	double denominator = 0;

	for (int i = 0; i < nOfClasses; i++) {

		denominator += exp(softmaxInputs[i]);

	}

	vector<double> answer(nOfClasses);

	for (int i = 0; i < nOfClasses; i++) {

		answer[i] = exp(softmaxInputs[i]) / denominator;

	}

	return answer;
}



//***************************************** Tensor transpose *******************


vector<vector<double>> CNN::tensorTranspose(vector<vector<double>> tensor) {

	int n = tensor.size();

	int m = tensor[0].size();

	for (int r = 0; r < n; r++) {

		int c1 = 0, c2 = m - 1;

		while (c1 < c2) {

			swap(tensor[r][c1], tensor[r][c2]);

			c1++;

			c2--;

		}

	}

	for (int c = 0; c < m; c++) {

		int r1 = 0, r2 = n - 1;

		while (r1 < r2) {

			swap(tensor[r1][c], tensor[r2][c]);

			r1++;

			r2--;

		}

	}

	return tensor;

}








//*********************************** Constructors *******************************

void CNN::parametersInit() {

	// Conv layers

	convWeights.resize(nOfConvLayers);

	convWeightsDiff.resize(nOfConvLayers);

	convBiases.resize(nOfConvLayers);

	convBiasesDiff.resize(nOfConvLayers);


	convWeightsV.resize(nOfConvLayers);

	convWeightsG.resize(nOfConvLayers);

	convBiasesV.resize(nOfConvLayers);

	convBiasesG.resize(nOfConvLayers);

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		convWeights[convLayer].resize(nOfFilters[convLayer]);

		convWeightsDiff[convLayer].resize(nOfFilters[convLayer]);

		convBiases[convLayer].resize(nOfFilters[convLayer]);

		convBiasesDiff[convLayer].resize(nOfFilters[convLayer]);


		convWeightsV[convLayer].resize(nOfFilters[convLayer]);

		convWeightsG[convLayer].resize(nOfFilters[convLayer]);

		convBiasesV[convLayer].resize(nOfFilters[convLayer]);

		convBiasesG[convLayer].resize(nOfFilters[convLayer]);

		for (int filter = 0; filter < nOfFilters[convLayer]; filter++) {

			if (convLayer == 0) {

				convWeights[convLayer][filter].resize(nOfImageChannels);

				convWeightsDiff[convLayer][filter].resize(nOfImageChannels);

				convWeightsV[convLayer][filter].resize(nOfImageChannels);

				convWeightsG[convLayer][filter].resize(nOfImageChannels);

			}

			else {

				convWeights[convLayer][filter].resize(nOfFilters[convLayer - 1]);

				convWeightsDiff[convLayer][filter].resize(nOfFilters[convLayer - 1]);

				convWeightsV[convLayer][filter].resize(nOfFilters[convLayer - 1]);

				convWeightsG[convLayer][filter].resize(nOfFilters[convLayer - 1]);

			}

			for (int k = 0; k < ((convLayer == 0) ? nOfImageChannels : nOfFilters[convLayer - 1]); k++) {

				convWeights[convLayer][filter][k].resize(sizeOfKernels[convLayer]);

				convWeightsDiff[convLayer][filter][k].resize(sizeOfKernels[convLayer]);

				convWeightsV[convLayer][filter][k].resize(sizeOfKernels[convLayer]);

				convWeightsG[convLayer][filter][k].resize(sizeOfKernels[convLayer]);

				for (int i = 0; i < sizeOfKernels[convLayer]; i++) {

					convWeights[convLayer][filter][k][i].resize(sizeOfKernels[convLayer]);

					convWeightsDiff[convLayer][filter][k][i].resize(sizeOfKernels[convLayer]);

					convWeightsV[convLayer][filter][k][i].resize(sizeOfKernels[convLayer]);

					convWeightsG[convLayer][filter][k][i].resize(sizeOfKernels[convLayer]);

					for (int j = 0; j < sizeOfKernels[convLayer]; j++) {

						int r = rand();

						double value = ((double)r / (double)RAND_MAX) / 100;

						convWeights[convLayer][filter][k][i][j] = value;

					}

				}

			}

			convBiases[convLayer][filter] = 0;

			convBiasesDiff[convLayer][filter] = 0;

		}

	}





	// Full layers

	fullWeights.resize(nOfFullLayers);

	fullWeightsDiff.resize(nOfFullLayers);

	fullBiases.resize(nOfFullLayers);

	fullBiasesDiff.resize(nOfFullLayers);


	fullWeightsV.resize(nOfFullLayers);

	fullWeightsG.resize(nOfFullLayers);

	fullBiasesV.resize(nOfFullLayers);

	fullBiasesG.resize(nOfFullLayers);

	for (int fullLayer = 0; fullLayer < nOfFullLayers; fullLayer++) {

		int nOfPrevLayerOutputSignals;

		if (fullLayer == 0) {

			nOfPrevLayerOutputSignals = (imageWidth / pow(sizeOfPooling, nOfConvLayers)) *
				(imageHeight / pow(sizeOfPooling, nOfConvLayers)) * 
				convWeights.back().size();

		}

		else {

			nOfPrevLayerOutputSignals = sizeOfLayers[fullLayer - 1];

		}

		fullWeights[fullLayer].resize(nOfPrevLayerOutputSignals);

		fullWeightsDiff[fullLayer].resize(nOfPrevLayerOutputSignals);

		fullBiases[fullLayer].resize(sizeOfLayers[fullLayer]);

		fullBiasesDiff[fullLayer].resize(sizeOfLayers[fullLayer]);


		fullWeightsV[fullLayer].resize(nOfPrevLayerOutputSignals);

		fullWeightsG[fullLayer].resize(nOfPrevLayerOutputSignals);

		fullBiasesV[fullLayer].resize(sizeOfLayers[fullLayer]);

		fullBiasesG[fullLayer].resize(sizeOfLayers[fullLayer]);

		for (int nOfOutput = 0; nOfOutput < nOfPrevLayerOutputSignals; nOfOutput++) {

			fullWeights[fullLayer][nOfOutput].resize(sizeOfLayers[fullLayer]);

			fullWeightsDiff[fullLayer][nOfOutput].resize(sizeOfLayers[fullLayer]);

			fullWeightsV[fullLayer][nOfOutput].resize(sizeOfLayers[fullLayer]);

			fullWeightsG[fullLayer][nOfOutput].resize(sizeOfLayers[fullLayer]);

			for (int nOfInput = 0; nOfInput < sizeOfLayers[fullLayer]; nOfInput++) {

				int r = rand();

				double value = (double)r / (double)RAND_MAX - 0.5;

				fullWeights[fullLayer][nOfOutput][nOfInput] = value;

				fullBiases[fullLayer][nOfInput] = 0;

			}

		}

	}




	// Output layer

	outputWeights.resize(sizeOfLayers.back());

	outputWeightsDiff.resize(sizeOfLayers.back());

	outputBiases.resize(nOfClasses);

	outputBiasesDiff.resize(nOfClasses);


	outputWeightsV.resize(sizeOfLayers.back());

	outputWeightsG.resize(sizeOfLayers.back());

	outputBiasesV.resize(nOfClasses);

	outputBiasesG.resize(nOfClasses);

	for (int nOfOutput = 0; nOfOutput < sizeOfLayers.back(); nOfOutput++) {

		outputWeights[nOfOutput].resize(nOfClasses);

		outputWeightsDiff[nOfOutput].resize(nOfClasses);

		outputWeightsV[nOfOutput].resize(nOfClasses);

		outputWeightsG[nOfOutput].resize(nOfClasses);

		for (int nOfInput = 0; nOfInput < nOfClasses; nOfInput++) {

			int r = rand();

			double value = (double)r / (double)RAND_MAX - 0.5;

			outputWeights[nOfOutput][nOfInput] = value;

			outputBiases[nOfInput] = 0;

		}

	}


}












void CNN::statesInit() {

	// Input image

	initialImage.resize(nOfImageChannels);

	for (int channel = 0; channel < nOfImageChannels; channel++) {

		initialImage[channel].resize(imageHeight);

		for (int row = 0; row < imageHeight; row++) {

			initialImage[channel][row].resize(imageWidth);

			for (int col = 0; col < initialImage[channel][row].size(); col++) {

				initialImage[channel][row][col] = (double)rand() / (double)RAND_MAX;

			}

		}

	}






	// Conv layers

	convLayersInputs.resize(nOfConvLayers);

	convLayersOutputs.resize(nOfConvLayers);

	convLayersDiff.resize(nOfConvLayers);

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		convLayersInputs[convLayer].resize(nOfFilters[convLayer]);

		convLayersOutputs[convLayer].resize(nOfFilters[convLayer]);

		convLayersDiff[convLayer].resize(nOfFilters[convLayer]);

		for (int map = 0; map < nOfFilters[convLayer]; map++) {

			int mapHeight = imageHeight / pow(sizeOfPooling, convLayer);

			int mapWidth = imageWidth / pow(sizeOfPooling, convLayer);

			convLayersInputs[convLayer][map].resize(mapHeight);

			convLayersOutputs[convLayer][map].resize(mapHeight);

			convLayersDiff[convLayer][map].resize(mapHeight);

			for (int row = 0; row < mapHeight; row++) {

				convLayersInputs[convLayer][map][row].resize(mapWidth);

				convLayersOutputs[convLayer][map][row].resize(mapWidth);

				convLayersDiff[convLayer][map][row].resize(mapWidth);

			}

		}

	}




	// Pooling layers

	poolingLayersOutputs.resize(nOfConvLayers);

	poolingLayersDiff.resize(nOfConvLayers);

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		poolingLayersOutputs[convLayer].resize(nOfFilters[convLayer]);

		poolingLayersDiff[convLayer].resize(nOfFilters[convLayer]);

		for (int map = 0; map < nOfFilters[convLayer]; map++) {

			int mapHeight = imageHeight / pow(sizeOfPooling, convLayer + 1);

			int mapWidth = imageWidth / pow(sizeOfPooling, convLayer + 1);

			poolingLayersOutputs[convLayer][map].resize(mapHeight);

			poolingLayersDiff[convLayer][map].resize(mapHeight);

			for (int row = 0; row < mapHeight; row++) {

				poolingLayersOutputs[convLayer][map][row].resize(mapWidth);

				poolingLayersDiff[convLayer][map][row].resize(mapWidth);

			}

		}

	}

	poolingMemory.resize(nOfConvLayers);

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		int nOfMaps = nOfFilters[convLayer];

		poolingMemory[convLayer].resize(nOfMaps);

		for (int map = 0; map < nOfMaps; map++) {

			int mapWidth = imageWidth / pow(sizeOfPooling, convLayer + 1);

			int mapHeight = imageHeight / pow(sizeOfPooling, convLayer + 1);

			poolingMemory[convLayer][map].resize(mapHeight);

			for (int row = 0; row < mapHeight; row++) {

				poolingMemory[convLayer][map][row].resize(mapWidth);

				for (int col = 0; col < mapWidth; col++) {

					poolingMemory[convLayer][map][row][col] = 0;

				}

			}

		}

	}





	// Full layers

	fullLayersInputs.resize(nOfFullLayers);

	fullLayersOutputs.resize(nOfFullLayers);

	fullLayersDiff.resize(nOfFullLayers);

	for (int fullLayer = 0; fullLayer < nOfFullLayers; fullLayer++) {

		fullLayersInputs[fullLayer].resize(sizeOfLayers[fullLayer]);

		fullLayersOutputs[fullLayer].resize(sizeOfLayers[fullLayer]);

		fullLayersDiff[fullLayer].resize(sizeOfLayers[fullLayer]);

	}



	// Output layers

	outputInputs.resize(nOfClasses);

	outputOutputs.resize(nOfClasses);

	outputDiff.resize(nOfClasses);

	

}














void CNN::init() {

	parametersInit();

	statesInit();

}







CNN::CNN() {

	init();

}












//*********************************** Predict *******************************

void CNN::forwardPropagation() {

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		int nOfPrevLayerMaps = (convLayer == 0) ? nOfImageChannels : nOfFilters[convLayer - 1];

		for (int map = 0; map < nOfFilters[convLayer]; map++) {

			for (int mapRow = 0; mapRow < convLayersInputs[convLayer][map].size(); mapRow++) {

				for (int mapCol = 0; mapCol < convLayersInputs[convLayer][map][mapRow].size();
					mapCol++) {

					// Calculate element

					double value = 0;

					for (int prevLayerMap = 0; prevLayerMap < nOfPrevLayerMaps; prevLayerMap++) {

						for (int filterRow = 0; filterRow < sizeOfKernels[convLayer]; 
							filterRow++) {

							for (int filterCol = 0; filterCol < sizeOfKernels[convLayer];
								filterCol++) {

								int rowOffset = filterRow - sizeOfKernels[convLayer] / 2;

								int colOffset = filterCol - sizeOfKernels[convLayer] / 2;

								int prevLayerMapRow = mapRow + rowOffset;

								int prevLayerMapCol = mapCol + colOffset;

								if (prevLayerMapRow < 0 || prevLayerMapCol < 0 || 
									prevLayerMapRow >= convLayersInputs[convLayer][map].size() ||
									prevLayerMapCol >= convLayersInputs[convLayer][map][mapRow].size()) {

									continue;

								}

								if (convLayer == 0) {

									value += initialImage[prevLayerMap][prevLayerMapRow][prevLayerMapCol] *
										convWeights[convLayer][map][prevLayerMap][filterRow][filterCol];

								}

								else {

									value += poolingLayersOutputs[convLayer - 1][prevLayerMap][prevLayerMapRow][prevLayerMapCol] * 
										convWeights[convLayer][map][prevLayerMap][filterRow][filterCol];

								}

								//value += 0.1; // For debug

							}

						}

					}

					value += convBiases[convLayer][map];

					convLayersInputs[convLayer][map][mapRow][mapCol] = value;

					convLayersOutputs[convLayer][map][mapRow][mapCol] =
						convActivation(convLayersInputs[convLayer][map][mapRow][mapCol]);

				}

			}

			// Pooling

			for (int poolRow = 0; poolRow < poolingLayersOutputs[convLayer][map].size(); poolRow++) {

				for (int poolCol = 0; poolCol < poolingLayersOutputs[convLayer][map][poolRow].size(); poolCol++) {

					// Pooling operation

					vector<double> values = {
						convLayersOutputs[convLayer][map][poolRow * 2][poolCol * 2],
						convLayersOutputs[convLayer][map][poolRow * 2][poolCol * 2 + 1],
						convLayersOutputs[convLayer][map][poolRow * 2 + 1][poolCol * 2],
						convLayersOutputs[convLayer][map][poolRow * 2 + 1][poolCol * 2 + 1]
					};

					double maximum = -LLONG_MIN;

					int index = 0;

					for (int i = 0; i < values.size(); i++) {

						if (maximum < values[i]) {

							maximum = values[i];

							index = i + 1;

						}

					}

					poolingLayersOutputs[convLayer][map][poolRow][poolCol] = maximum;

					poolingMemory[convLayer][map][poolRow][poolCol] = index;

				}

			}

		}

	}


	// Full layers

	for (int nOfInput = 0; nOfInput < fullLayersInputs[0].size(); nOfInput++) {

		double value = fullBiases[0][nOfInput];

		int nOfOutput = 0;

		for (int map = 0; map < poolingLayersOutputs.back().size(); map++) {

			for (int row = 0; row < poolingLayersOutputs.back()[map].size(); row++) {

				for (int col = 0; col < poolingLayersOutputs.back()[map][row].size(); col++) {

					// Calculate input value

					value += fullWeights[0][nOfOutput][nOfInput] *
						poolingLayersOutputs.back()[map][row][col];

					nOfOutput++;

				}

			}

		}

		fullLayersInputs[0][nOfInput] = value;

		fullLayersOutputs[0][nOfInput] = layerActivation(value);

	}

	for (int nOfFullLayer = 1; nOfFullLayer < nOfFullLayers; nOfFullLayer++) {

		for (int nOfInput = 0; nOfInput < sizeOfLayers[nOfFullLayer]; nOfInput++) {

			double value = fullBiases[nOfFullLayer][nOfInput];

			for (int nOfOutput = 0; nOfOutput < sizeOfLayers[nOfFullLayer - 1]; nOfOutput++) {

				value += fullLayersOutputs[nOfFullLayer - 1][nOfOutput] *
					fullWeights[nOfFullLayer][nOfOutput][nOfInput];

			}

			fullLayersInputs[nOfFullLayer][nOfInput] = value;

			fullLayersOutputs[nOfFullLayer][nOfInput] = layerActivation(value);

		}

	}

	// Output layer

	for (int nOfInput = 0; nOfInput < nOfClasses; nOfInput++) {

		double value = outputBiases[nOfInput];

		for (int nOfOutput = 0; nOfOutput < sizeOfLayers.back(); nOfOutput++) {

			value += fullLayersOutputs.back()[nOfOutput] * outputWeights[nOfOutput][nOfInput];

		}

		outputInputs[nOfInput] = value;

	}

	outputOutputs = softmaxOutputs(outputInputs);

	lossValue = -log(outputOutputs[targetClass]);

}



vector<double> CNN::predict(vector<vector<vector<double>>> image) {

	if (image.size() != nOfImageChannels) {

		return { -1 };

	}

	for (int channel = 0; channel < nOfImageChannels; channel++) {

		if (image[channel].size() != imageHeight) {

			return { -1 };

		}

		for (int row = 0; row < imageHeight; row++) {

			if (image[channel][row].size() != imageWidth) {

				return { -1 };

			}

			for (int col = 0; col < imageWidth; col++) {

				initialImage[channel][row][col] = image[channel][row][col];

			}

		}

	}

	forwardPropagation();

	return outputOutputs;

}







//*********************************** Train *******************************

void CNN::backPropagation() {

	lossValue = -log(outputOutputs[targetClass]);

	// Output layer

	for (int nOfOutputNeuron = 0; nOfOutputNeuron < nOfClasses; nOfOutputNeuron++) {

		outputDiff[nOfOutputNeuron] = outputOutputs[nOfOutputNeuron];

		if (nOfOutputNeuron == targetClass) {

			outputDiff[nOfOutputNeuron] -= 1.0;

		}

		outputBiasesDiff[nOfOutputNeuron] = outputDiff[nOfOutputNeuron];

		for (int prevLayerNeuron = 0; prevLayerNeuron < sizeOfLayers.back(); prevLayerNeuron++) {

			outputWeightsDiff[prevLayerNeuron][nOfOutputNeuron] = outputDiff[nOfOutputNeuron] *
				fullLayersOutputs.back()[prevLayerNeuron];

		}

	}

	// Full layers

	for (int fullLayer = sizeOfLayers.size() - 1; fullLayer >= 0; fullLayer--) {

		int nOfNextLayerNeurons;

		if (fullLayer == sizeOfLayers.size() - 1) {

			nOfNextLayerNeurons = nOfClasses;

		}

		else {

			nOfNextLayerNeurons = fullLayersDiff[fullLayer + 1].size();

		}

		for (int currNeuron = 0; currNeuron < fullLayersDiff[fullLayer].size(); currNeuron++) {

			double value = 0;

			for (int nextNeuron = 0; nextNeuron < nOfNextLayerNeurons; nextNeuron++) {

				if (fullLayer == sizeOfLayers.size() - 1) {

					value += outputDiff[nextNeuron] * outputWeights[currNeuron][nextNeuron];

				}

				else {

					value += fullLayersDiff[fullLayer + 1][nextNeuron] *
						fullWeights[fullLayer + 1][currNeuron][nextNeuron];

				}

			}

			fullLayersDiff[fullLayer][currNeuron] = value * 
				layerActivationDiff(fullLayersInputs[fullLayer][currNeuron]);

			fullBiasesDiff[fullLayer][currNeuron] = fullLayersDiff[fullLayer][currNeuron];

			if (fullLayer > 0) {

				for (int prevNeuron = 0; prevNeuron < fullLayersDiff[fullLayer - 1].size(); prevNeuron++) {

					fullWeightsDiff[fullLayer][prevNeuron][currNeuron] =
						fullLayersDiff[fullLayer][currNeuron] *
						fullLayersOutputs[fullLayer - 1][prevNeuron];

				}

			}

		}

	}

	int nOfPoolingOutNeuron = 0;

	for (int map = 0; map < poolingLayersOutputs.back().size(); map++) {

		for (int row = 0; row < poolingLayersOutputs.back()[map].size(); row++) {

			for (int col = 0; col < poolingLayersOutputs.back()[map][row].size(); col++) {

				for (int fullNeuron = 0; fullNeuron < fullLayersDiff[0].size(); fullNeuron++) {

					fullWeightsDiff[0][nOfPoolingOutNeuron][fullNeuron] =
						fullLayersDiff[0][fullNeuron] *
						poolingLayersOutputs.back()[map][row][col];

				}

				nOfPoolingOutNeuron++;

			}

		}

	}

	// Pooling and convolution layers

	for (int convPoolingLayer = nOfConvLayers - 1; convPoolingLayer >= 0; convPoolingLayer--) {

		// Zero conv layers diff

		for (int map = 0; map < convLayersDiff[convPoolingLayer].size(); map++) {

			for (int row = 0; row < convLayersDiff[convPoolingLayer][map].size(); row++) {

				for (int col = 0; col < convLayersDiff[convPoolingLayer][map][row].size(); col++) {

					convLayersDiff[convPoolingLayer][map][row][col] = 0;

				}

			}

		}

		// Pooling and conv layer

		int currNeuron = 0;

		for (int map = 0; map < poolingLayersOutputs[convPoolingLayer].size(); map++) {

			for (int row = 0; row < poolingLayersOutputs[convPoolingLayer][map].size(); row++) {

				for (int col = 0; col < poolingLayersOutputs[convPoolingLayer][map][row].size(); col++) {

					if (convPoolingLayer == nOfConvLayers - 1) {

						double value = 0;

						for (int nextNeuron = 0; nextNeuron < fullLayersDiff[0].size(); nextNeuron++) {

							value += fullLayersDiff[0][nextNeuron] *
								fullWeights[0][currNeuron][nextNeuron];

						}

						poolingLayersDiff[convPoolingLayer][map][row][col] = value;

					}

					else {

						double value = 0;

						for (int nextMap = 0; nextMap < convLayersDiff[convPoolingLayer + 1].size(); nextMap++) {

							vector<vector<double>> filter = convWeights[convPoolingLayer + 1][nextMap][map];

							filter = tensorTranspose(filter);

							for (int filterRow = 0; filterRow < filter.size(); filterRow++) {

								for (int filterCol = 0; filterCol < filter[filterRow].size(); filterCol++) {

									int rowOffset = filterRow - filter.size() / 2;

									int colOffset = filterCol - filter[filterRow].size() / 2;

									if (row + rowOffset < 0 || row + rowOffset >= poolingLayersOutputs[convPoolingLayer][map].size() ||
										col + colOffset < 0 || col + colOffset >= poolingLayersOutputs[convPoolingLayer][map][row].size()) {

										continue;

									}

									/*value += convLayersDiff[convPoolingLayer + 1][nextMap][row + rowOffset][col + colOffset] *
										convWeights[convPoolingLayer + 1][nextMap][map][filterRow][filterCol];*/

									value += convLayersDiff[convPoolingLayer + 1][nextMap][row + rowOffset][col + colOffset] *
										filter[filterRow][filterCol];

								}

							}

						}

						poolingLayersDiff[convPoolingLayer][map][row][col] = value;

					}

					currNeuron++;

					int rowPoolingOffset = 0;

					int colPoolingOffset = 0;

					int poolMemory = poolingMemory[convPoolingLayer][map][row][col];

					if (poolMemory > 2) {

						rowPoolingOffset = 1;

					}

					if (poolMemory % 2 == 0) {

						colPoolingOffset = 1;

					}

					convLayersDiff[convPoolingLayer][map][row * 2 + rowPoolingOffset][col * 2 + colPoolingOffset] =
						poolingLayersDiff[convPoolingLayer][map][row][col] *
						convActivationDiff(convLayersInputs[convPoolingLayer][map][row * 2 + rowPoolingOffset][col * 2 + colPoolingOffset]);

				}

			}

		}

		// Zero conv weights gradients

		for (int map = 0; map < convWeights[convPoolingLayer].size(); map++) {

			convBiasesDiff[convPoolingLayer][map] = 0;

			for (int prevMap = 0; prevMap < convWeights[convPoolingLayer][map].size(); prevMap++) {

				for (int row = 0; row < convWeights[convPoolingLayer][map][prevMap].size(); row++) {

					for (int col = 0; col < convWeights[convPoolingLayer][map][prevMap][row].size(); col++) {

						convWeightsDiff[convPoolingLayer][map][prevMap][row][col] = 0;

					}

				}

			}

		}

		// Calc conv weights gradients

		for (int currMap = 0; currMap < convLayersDiff[convPoolingLayer].size(); currMap++) {

			for (int currRow = 0; currRow < convLayersDiff[convPoolingLayer][currMap].size(); currRow++) {

				for (int currCol = 0; currCol < convLayersDiff[convPoolingLayer][currMap][currRow].size(); currCol++) {

					convBiasesDiff[convPoolingLayer][currMap] +=
						convLayersDiff[convPoolingLayer][currMap][currRow][currCol];

					for (int filterMap = 0; filterMap < convWeightsDiff[convPoolingLayer][currMap].size(); filterMap++) {

						for (int filterRow = 0; filterRow < convWeightsDiff[convPoolingLayer][currMap][filterMap].size(); filterRow++) {

							for (int filterCol = 0; filterCol < convWeightsDiff[convPoolingLayer][currMap][filterMap][filterRow].size(); filterCol++) {

								int rowOffset = filterRow - convWeightsDiff[convPoolingLayer][currMap][filterMap].size() / 2;

								int colOffset = filterCol - convWeightsDiff[convPoolingLayer][currMap][filterMap][filterRow].size() / 2;

								if (currRow + rowOffset < 0 ||
									currRow + rowOffset >= convLayersDiff[convPoolingLayer][currMap].size() ||
									currCol + colOffset < 0 ||
									currCol + colOffset >= convLayersDiff[convPoolingLayer][currMap][currRow].size()) {

									continue;

								}

								if (convPoolingLayer > 0) {

									convWeightsDiff[convPoolingLayer][currMap][filterMap][filterRow][filterCol] +=
										convLayersDiff[convPoolingLayer][currMap][currRow][currCol] *
										poolingLayersOutputs[convPoolingLayer - 1][filterMap][currRow + rowOffset][currCol + colOffset];

								}

								else {

									convWeightsDiff[convPoolingLayer][currMap][filterMap][filterRow][filterCol] +=
										convLayersDiff[convPoolingLayer][currMap][currRow][currCol] *
										initialImage[filterMap][currRow + rowOffset][currCol + colOffset];

								}

							}

						}

					}

				}

			}

		}

	}

}


void CNN::updateWeights(bool useAdaptive) {

	// Conv layers

	for (int layer = 0; layer < convWeights.size(); layer++) {

		for (int filter = 0; filter < convWeights[layer].size(); filter++) {

			convBiasesV[layer][filter] = beta1 * convBiasesV[layer][filter] +
				(1 - beta1) * convBiasesDiff[layer][filter];

			convBiasesG[layer][filter] = beta2 * convBiasesG[layer][filter] +
				(1 - beta2) * pow(convBiasesDiff[layer][filter], 2);

			convBiases[layer][filter] -= learningRate * convBiasesV[layer][filter] / 
				(1.0 + (double)useAdaptive * (-1.0 + sqrt(convBiasesG[layer][filter] + epsilon)));

			for (int map = 0; map < convWeights[layer][filter].size(); map++) {

				for (int row = 0; row < convWeights[layer][filter][map].size(); row++) {

					for (int col = 0; col < convWeights[layer][filter][map][row].size(); col++) {

						convWeightsV[layer][filter][map][row][col] = beta1 * convWeightsV[layer][filter][map][row][col] + 
							(1 - beta1) * convWeightsDiff[layer][filter][map][row][col];

						convWeightsG[layer][filter][map][row][col] = beta2 * convWeightsG[layer][filter][map][row][col] +
							(1 - beta2) * pow(convWeightsDiff[layer][filter][map][row][col], 2);

						convWeights[layer][filter][map][row][col] -= learningRate * convWeightsV[layer][filter][map][row][col] / 
							(1.0 + (double)useAdaptive * (-1.0 + sqrt(convWeightsG[layer][filter][map][row][col] + epsilon)));

					}

				}

			}

		}

	}

	// Full layers

	for (int layer = 0; layer < fullWeights.size(); layer++) {

		for (int prev = 0; prev < fullWeights[layer].size(); prev++) {

			for (int curr = 0; curr < fullWeights[layer][prev].size(); curr++) {

				fullWeightsV[layer][prev][curr] = beta1 * fullWeightsV[layer][prev][curr] +
					(1 - beta1) * fullWeightsDiff[layer][prev][curr];

				fullWeightsG[layer][prev][curr] = beta2 * fullWeightsG[layer][prev][curr] +
					(1 - beta2) * pow(fullWeightsDiff[layer][prev][curr], 2);

				fullWeights[layer][prev][curr] -= learningRate * fullWeightsV[layer][prev][curr] / 
					(1.0 + (double)useAdaptive * (-1.0 + sqrt(fullWeightsG[layer][prev][curr] + epsilon)));

			}

		}

	}

	for (int layer = 0; layer < fullBiases.size(); layer++) {

		for (int curr = 0; curr < fullBiases[layer].size(); curr++) {

			fullBiasesV[layer][curr] = beta1 * fullBiasesV[layer][curr] +
				(1 - beta1) * fullBiasesDiff[layer][curr];

			fullBiasesG[layer][curr] = beta2 * fullBiasesG[layer][curr] +
				(1 - beta2) * pow(fullBiasesDiff[layer][curr], 2);

			fullBiases[layer][curr] -= learningRate * fullBiasesV[layer][curr] /
				(1.0 + (double)useAdaptive * (-1.0 + sqrt(fullBiasesG[layer][curr] + epsilon)));

		}

	}

	// Output layer

	for (int prev = 0; prev < outputWeights.size(); prev++) {

		for (int output = 0; output < outputWeights[prev].size(); output++) {

			outputWeightsV[prev][output] = beta1 * outputWeightsV[prev][output] +
				(1 - beta1) * outputWeightsDiff[prev][output];

			outputWeightsG[prev][output] = beta2 * outputWeightsG[prev][output] +
				(1 - beta2) * pow(outputWeightsDiff[prev][output], 2);

			outputWeights[prev][output] -= learningRate * outputWeightsV[prev][output] /
				(1.0 - (double)useAdaptive * (-1.0 * sqrt(outputWeightsG[prev][output] + epsilon)));

		}

	}

	for (int output = 0; output < outputBiases.size(); output++) {

		outputBiasesV[output] = beta1 * outputBiasesV[output] +
			(1 - beta1) * outputBiasesDiff[output];

		outputBiasesG[output] = beta2 * outputBiasesG[output] +
			(1 - beta2) * pow(outputBiasesDiff[output], 2);

		outputBiases[output] -= learningRate * outputBiasesV[output] / 
			(1.0 + (double)useAdaptive * (-1.0 * sqrt(outputBiasesG[output] + epsilon)));

	}

}

double CNN::fit(vector<vector<vector<double>>> image, int target, bool useAdaptive = false) {

	if (target < 0 || target >= nOfClasses) {

		return -1;

	}

	targetClass = target;

	if (image.size() != nOfImageChannels) {

		return -1;

	}

	for (int channel = 0; channel < nOfImageChannels; channel++) {

		if (image[channel].size() != imageHeight) {

			return -1;

		}

		for (int row = 0; row < imageHeight; row++) {

			if (image[channel][row].size() != imageWidth) {

				return -1;

			}

			for (int col = 0; col < imageWidth; col++) {

				initialImage[channel][row][col] = image[channel][row][col];

			}

		}

	}

	forwardPropagation();

	backPropagation();

	updateWeights(useAdaptive);

	return lossValue;

}

double CNN::getLoss(vector<vector<vector<double>>> image, int target) {

	if (target < 0 || target >= nOfClasses) {

		return -1;

	}

	targetClass = target;

	if (image.size() != nOfImageChannels) {

		return -1;

	}

	for (int channel = 0; channel < nOfImageChannels; channel++) {

		if (image[channel].size() != imageHeight) {

			return -1;

		}

		for (int row = 0; row < imageHeight; row++) {

			if (image[channel][row].size() != imageWidth) {

				return -1;

			}

			for (int col = 0; col < imageWidth; col++) {

				initialImage[channel][row][col] = image[channel][row][col];

			}

		}

	}

	forwardPropagation();

	return lossValue;

}










//*********************************** Save/load *********************************

bool CNN::save(string fileName) {

	ofstream fileOut(fileName);

	if (!fileOut.is_open()) {

		return false;

	}

	// Conv layers

	for (int layer = 0; layer < convWeights.size(); layer++) {

		for (int filter = 0; filter < convWeights[layer].size(); filter++) {

			fileOut << convBiases[layer][filter] << " ";

			for (int map = 0; map < convWeights[layer][filter].size(); map++) {

				for (int row = 0; row < convWeights[layer][filter][map].size(); row++) {

					for (int col = 0; col < convWeights[layer][filter][map][row].size(); col++) {

						fileOut << convWeights[layer][filter][map][row][col] << " ";

					}

				}

			}

		}

	}

	// Full layers

	for (int layer = 0; layer < fullWeights.size(); layer++) {

		for (int prev = 0; prev < fullWeights[layer].size(); prev++) {

			for (int curr = 0; curr < fullWeights[layer][prev].size(); curr++) {

				fileOut << fullWeights[layer][prev][curr] << " ";

			}

		}

	}

	for (int layer = 0; layer < fullBiases.size(); layer++) {

		for (int curr = 0; curr < fullBiases[layer].size(); curr++) {

			fileOut << fullBiases[layer][curr] << " ";

		}

	}

	// Output layer

	for (int prev = 0; prev < outputWeights.size(); prev++) {

		for (int output = 0; output < outputWeights[prev].size(); output++) {

			fileOut << outputWeights[prev][output] << " ";

		}

	}

	for (int output = 0; output < outputBiases.size(); output++) {

		fileOut << outputBiases[output] << " ";

	}

	fileOut.close();

	return true;

}

bool CNN::load(string fileName) {

	ifstream fileIn(fileName);

	if (!fileIn.is_open()) {

		return false;

	}

	// Conv layers

	for (int layer = 0; layer < convWeights.size(); layer++) {

		for (int filter = 0; filter < convWeights[layer].size(); filter++) {

			fileIn >> convBiases[layer][filter];

			for (int map = 0; map < convWeights[layer][filter].size(); map++) {

				for (int row = 0; row < convWeights[layer][filter][map].size(); row++) {

					for (int col = 0; col < convWeights[layer][filter][map][row].size(); col++) {

						fileIn >> convWeights[layer][filter][map][row][col];

					}

				}

			}

		}

	}

	// Full layers

	for (int layer = 0; layer < fullWeights.size(); layer++) {

		for (int prev = 0; prev < fullWeights[layer].size(); prev++) {

			for (int curr = 0; curr < fullWeights[layer][prev].size(); curr++) {

				fileIn >> fullWeights[layer][prev][curr];

			}

		}

	}

	for (int layer = 0; layer < fullBiases.size(); layer++) {

		for (int curr = 0; curr < fullBiases[layer].size(); curr++) {

			fileIn >> fullBiases[layer][curr];

		}

	}

	// Output layer

	for (int prev = 0; prev < outputWeights.size(); prev++) {

		for (int output = 0; output < outputWeights[prev].size(); output++) {

			fileIn >> outputWeights[prev][output];

		}

	}

	for (int output = 0; output < outputBiases.size(); output++) {

		fileIn >> outputBiases[output];

	}

	fileIn.close();

	return true;

}