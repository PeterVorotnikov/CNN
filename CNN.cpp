#include "CNN.h"




//***************************************** Activation functions *******************


double CNN::convActivation(double x) {

	if (x > 0) {

		return x;

	}

	else {

		return 0;

	}
}

double CNN::convActivationDiff(double x) {

	if (x > 0) {

		return 1;

	}

	else {

		return 0;

	}
}











double CNN::layerActivation(double x) {

	return 1.0 / (1.0 + exp(-x));

}
double CNN::layerActivationDiff(double x) {

	return layerActivation(x) * (1 - layerActivation(x));

}











vector<double> CNN::softmaxOutputs(vector<double> softmaxInputs, int nOfClass) {

	double denominator = 0;

	for (int i = 0; i < nOfClasses; i++) {

		denominator += exp(softmaxInputs[i]);

	}

	vector<double> answer(nOfClasses);

	for (int i = 0; i < nOfClasses; i++) {

		answer[i] = exp(softmaxInputs[i] / denominator);

	}

	return answer;
}











//*********************************** Constructors *******************************

void CNN::parametersInit() {

	// Conv layers

	convWeights.resize(nOfConvLayers);

	convWeightsDiff.resize(nOfConvLayers);

	convBiases.resize(nOfConvLayers);

	convBiasesDiff.resize(nOfConvLayers);

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		convWeights[convLayer].resize(nOfFilters[convLayer]);

		convWeightsDiff[convLayer].resize(nOfFilters[convLayer]);

		convBiases[convLayer].resize(nOfFilters[convLayer]);

		convBiasesDiff[convLayer].resize(nOfFilters[convLayer]);

		for (int filter = 0; filter < nOfFilters[convLayer]; filter++) {

			if (convLayer == 0) {

				convWeights[convLayer][filter].resize(1);

				convWeightsDiff[convLayer][filter].resize(1);

			}

			else {

				convWeights[convLayer][filter].resize(nOfFilters[convLayer - 1]);

				convWeightsDiff[convLayer][filter].resize(nOfFilters[convLayer - 1]);

			}

			for (int k = 0; k < ((convLayer == 0) ? 1 : nOfFilters[convLayer - 1]); k++) {

				convWeights[convLayer][filter][k].resize(sizeOfKernels[convLayer]);

				convWeightsDiff[convLayer][filter][k].resize(sizeOfKernels[convLayer]);

				for (int i = 0; i < sizeOfKernels[convLayer]; i++) {

					convWeights[convLayer][filter][k][i].resize(sizeOfKernels[convLayer]);

					convWeightsDiff[convLayer][filter][k][i].resize(sizeOfKernels[convLayer]);

					for (int j = 0; j < sizeOfKernels[convLayer]; j++) {

						int r = rand();

						double value = (double)r / (double)RAND_MAX - 0.5;

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

		for (int nOfOutput = 0; nOfOutput < nOfPrevLayerOutputSignals; nOfOutput++) {

			fullWeights[fullLayer][nOfOutput].resize(sizeOfLayers[fullLayer]);

			fullWeightsDiff[fullLayer][nOfOutput].resize(sizeOfLayers[fullLayer]);

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

	for (int nOfOutput = 0; nOfOutput < sizeOfLayers.back(); nOfOutput++) {

		outputWeights[nOfOutput].resize(nOfClasses);

		outputWeightsDiff[nOfOutput].resize(nOfClasses);

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

	for (int convLayer = 0; convLayer < nOfConvLayers; convLayer++) {

		poolingLayersOutputs[convLayer].resize(nOfFilters[convLayer]);

		for (int map = 0; map < nOfFilters[convLayer]; map++) {

			int mapHeight = imageHeight / pow(sizeOfPooling, convLayer + 1);

			int mapWidth = imageWidth / pow(sizeOfPooling, convLayer + 1);

			poolingLayersOutputs[convLayer][map].resize(mapHeight);

			for (int row = 0; row < mapHeight; row++) {

				poolingLayersOutputs[convLayer][map][row].resize(mapWidth);

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

void CNN::forwardPropagationConvLayers() {

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

								value++;

							}

						}

					}

					value += convBiases[convLayer][map];

					convLayersInputs[convLayer][map][mapRow][mapCol] = value;

					convLayersOutputs[convLayer][map][mapRow][mapCol] =
						convActivation(convLayersInputs[convLayer][map][mapRow][mapCol]);

				}

			}

		}

	}

}