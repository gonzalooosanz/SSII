package ssii.rna;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class BasicANNForLogicalAND {
	
	public static double AND_IDEAL[][] = {{ 0.0 },{ 0.0 },
			{ 0.0 },{ 1.0 } };
	public static double AND_INPUT[][] = {{ 0.0, 0.0 },{ 1.0, 0.0 },
			{ 0.0, 1.0 },{ 1.0, 1.0 } };



	private static void testNetworkwithTrainingExamples(BasicNetwork network,
			NeuralDataSet trainingSet) {
		System.out.println("Neural Network Results:");
		for(MLDataPair pair: trainingSet ) {
			final MLData output = network.compute(pair.getInput());
			System.out.println(pair.getInput().getData(0) + "," + 
			pair.getInput().getData(1) + ", actual=" + output.getData(0) 
			+ ",ideal=" +pair.getIdeal().getData(0));

		}
	}

	private static NeuralDataSet trainANN(BasicNetwork network) {
		NeuralDataSet trainingSet = new
				BasicNeuralDataSet(AND_INPUT, AND_IDEAL);
		final Train train = new ResilientPropagation(network,
				trainingSet);

		int epoch = 1;
		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:"
					+ train.getError());
			epoch++;
		} while(train.getError() > 0.0001);
		return trainingSet;
	}

	private static BasicNetwork createANN() {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true,2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true,4));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true,1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	
	/** 
	 * Given a trained network, it applies this network to recognise an a particular input
	 * It returns the activation of the network
	 * 
	 * @param network a trained ANN
	 * @param imagePath the file to recognise
	 * @return the activation of the network
	 */
	private static MLData classifyOneInput(BasicNetwork network, double[] inputData) {
		MLData input=new org.encog.neural.data.basic.BasicNeuralData(inputData);
		MLData output = network.compute(input);		
		System.out.println(inputData[0] + "," + 
				inputData[1] + ", result=" + output.getData(0));		
		return output;
	}
	
	public static void main( String[] args )
	{
		BasicNetwork network = createANN();
		NeuralDataSet trainingSet = trainANN(network);
		// this part is not really necessary and it is included only to show
		// how the ANN performs with known examples
		testNetworkwithTrainingExamples(network, trainingSet);
		// classifying a sample input
		classifyOneInput(network, new double[] {0.0,1.0});
	}
}
