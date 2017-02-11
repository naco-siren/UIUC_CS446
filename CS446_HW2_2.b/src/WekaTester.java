import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class WekaTester {

    public static void main(String[] args) throws Exception {

//		if (args.length != 1) {
//			System.err.println("Usage: WekaTester arff-file");
//			System.exit(-1);
//		}
//		// Load the data
//		Instances data = new Instances(new FileReader(new File(args[0])));

		String arffFileName = "./badges.arff";
		Instances data = new Instances(new FileReader(new File(arffFileName)));


		// The last attribute is the class label
		data.setClassIndex(data.numAttributes() - 1);

		// Train on 80% of the data and test on 20%
		Instances train = data.trainCV(5, 0);
		Instances test = data.testCV(5, 0);

		// Create a new ID3 classifier. This is the modified one where you can
		// set the depth of the tree.
		Id3 classifier = new Id3();

		// An example depth. If this value is -1, then the tree is grown to full depth.
		// TODO: Change depth here
		classifier.setMaxDepth(4);

		// Train
		classifier.buildClassifier(train);

		// Print the classfier
		System.out.println(classifier);
		System.out.println();

		// Evaluate on the test set
		Evaluation evaluation = new Evaluation(test);
		evaluation.evaluateModel(classifier, test);
		System.out.println(evaluation.toSummaryString());

    }
}
