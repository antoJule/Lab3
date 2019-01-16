package hr.nrp.lab.test.main;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Glavna {
	public double String XOR_INPUT[][] = { {"vine", "wino","vin" }, { "sandwich", "sandvitx","szendvics" }};
			
	
	public static String XOR_IDEAL[][] = { { "vino" }, { "sendvic" } };
	
	public static void main(String... args) {
		
		BasicNetwork network= new BasicNetwork();
		network.addLayer(new BasicLayer(null,true,2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT,XOR_IDEAL);
		
		final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		
		int epoch=1;
		
		
		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		} while(train.getError() > 0.01);
		train.finishTraining();
 
		// test the neural network
		System.out.println("Neural Network Results:");
		for(MLDataPair pair: trainingSet ) {
			final MLData output = network.compute(pair.getInput());
			System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
					+ ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
		}
 
		Encog.getInstance().shutdown();
	}
		

}
