import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;

/**
 * Created by nacos on 2016/9/20.
 */
public class ReportingModule {

    public static void do5FoldCrossValidation() throws Exception {
        // Get Results
        double[][] accuracies = new double[5][5];
        accuracies[0] = do_5FCV_on_SGD();
        accuracies[1] = do_5FCV_on_ID3(-1);
        accuracies[2] = do_5FCV_on_ID3(4);
        accuracies[3] = do_5FCV_on_ID3(8);
        accuracies[4] = do_5FCV_on_Stumps();
        double[] avgAccuracies = new double[5];
        double[] stdDevs = new double[5];
        double[][] confidenceIntervals = new double[5][2];

        // Present every algorithm's stat
        for(int i = 0; i < 5; i++){
            avgAccuracies[i] = average(accuracies[i]);
            stdDevs[i] = standardDeviation(accuracies[i]);
            confidenceIntervals[i] = confidenceInterval(accuracies[i]);

            System.out.println("\n=== 2.1.(" + (char)('a' + i) + ") performance ===");
            System.out.println("Average: " + avgAccuracies[i]);
            System.out.println("Std. deviation: " + stdDevs[i]);
            System.out.println("Confidence interval: (" + confidenceIntervals[i][0] + ", " + confidenceIntervals[i][1] + ")");
        }

        // Calculate t-distribution
        double[][] pValues = new double[5][5];
        for(int i = 0; i < 5; i++){
            for(int j = 0; j < i; j++){
                pValues[i][j] =
                        (avgAccuracies[i] - avgAccuracies[j]) /
                                Math.sqrt(
                                        (4.0 * Math.pow(stdDevs[i], 2.0) + 4.0 * Math.pow(stdDevs[j], 2.0)) /
                                                (5.0 + 5.0 - 2.0) * (1.0 / 5.0 + 1.0 / 5.0));
                System.out.print(pValues[i][j] + " ");
            }
            System.out.print("\n");
        }

        return;
    }

    /**
     * 2.1.(a)
     */
    public static double[] do_5FCV_on_SGD() throws Exception{
        double[] accuracies = new double[5];

        // Read .ARFF files
        String badgeSrcFile = "./badges/badges.modified.data.fold";
        ArffDataset[] datas = new ArffDataset[5];
        for(int i = 0; i < 5; i++){
            datas[i] = ArffDataset.readFromFile(badgeSrcFile + (i+1) + ".arff");
        }

        // 5-fold cross validation
        for(int i = 0; i < 5; i++){
            //System.out.println("===== Testing the " + (i+1) + "-th fold =====");
            //ArffDataset totalArffDataset = ArffDataset.combineDatasets(datas);

            // Prepare training dataset and test dataset
            ArffDataset testDataset = datas[i];
            ArffDataset[] trainDatasets = new ArffDataset[4];
            int tempIndex = 0;
            for(int j = 0; j < 5; j++){
                if(j != i){
                    trainDatasets[tempIndex++] = datas[j];
                }
            }
            ArffDataset trainDataset = ArffDataset.combineDatasets(trainDatasets);

            // Train
            StochasticGradientDescent sgd = new StochasticGradientDescent(trainDataset);
            sgd.buildClassifier();

            // Evaluate on the test set
            accuracies[i] = sgd.accuracyRateOn(testDataset);
        }

        return accuracies;
    }

    /**
     * 2.1.(b)(c)(d)
     */
    public static double[] do_5FCV_on_ID3(int depth) throws Exception{
        double[] accuracies = new double[5];

        // Read .ARFF files
        String badgeSrcFile = "./badges/badges.modified.data.fold";
        Instances[] datas = new Instances[5];
        for(int i = 0; i < 5; i++){
            datas[i] = new Instances(new FileReader(new File(badgeSrcFile + (i+1) + ".arff")));
            datas[i].setClassIndex(datas[i].numAttributes() - 1);
        }

        // 5-fold cross validation
        for(int i = 0; i < 5; i++){
            //System.out.println("===== Testing the " + (i+1) + "-th fold =====");

            // Prepare training dataset and test dataset
            Instances testDataset = datas[i];
            Instances trainDataSet = new Instances(datas[i], 237);
            for(int j = 0; j < 5; j++){
                if(j != i){
                    for(int k = 0; k < datas[j].numInstances(); k++) {
                        trainDataSet.add(datas[j].instance(k));
                    }
                }
            }

            // Use ID3 to build classifier with given depth
            Id3 classifier = new Id3();
            classifier.setMaxDepth(depth);
            classifier.buildClassifier(trainDataSet);
            System.out.println(classifier);

            // Evaluate on the test set
            Evaluation evaluation = new Evaluation(testDataset);
            evaluation.evaluateModel(classifier, testDataset);
            System.out.println(evaluation.toSummaryString());

            //System.out.println(evaluation.toSummaryString());
            accuracies[i] = 1 - evaluation.errorRate();
        }

        return accuracies;
    }

    /**
     * 2.1.(e)
     */
    public static double[] do_5FCV_on_Stumps() throws Exception{
        double[] accuracies = new double[5];

        // Read .ARFF files
        String badgeSrcFile = "./badges/badges.modified.data.fold";
        Instances[] datas = new Instances[5];
        for(int i = 0; i < 5; i++){
            datas[i] = new Instances(new FileReader(new File(badgeSrcFile + (i+1) + ".arff")));
            datas[i].setClassIndex(datas[i].numAttributes() - 1);
        }

        // 5-fold cross validation
        for(int i = 0; i < 5; i++){
            //System.out.println("===== Testing the " + (i+1) + "-th fold =====");

            // Prepare training dataset and test dataset
            Instances testDataset = datas[i];
            Instances trainDataSet = new Instances(datas[i], 237);
            for(int j = 0; j < 5; j++){
                if(j != i){
                    for(int k = 0; k < datas[j].numInstances(); k++) {
                        trainDataSet.add(datas[j].instance(k));
                    }
                }
            }

            // Train stumps over trainDataset
            // and get new stump featured trainDataset and testDataset
            StumpAsFeature stumpAsFeature = new StumpAsFeature(trainDataSet, testDataset);
            ArffDataset newTrainDataset = stumpAsFeature.getStumpFeaturedTrainDataset();
            ArffDataset newTestDataset = stumpAsFeature.getStumpFeaturedTestDataset();


            // Train using SGD
            StochasticGradientDescent sgd = new StochasticGradientDescent(newTrainDataset);
            sgd.buildClassifier();

            // Evaluate on the test set
            accuracies[i] = sgd.accuracyRateOn(newTestDataset);
        }

        return accuracies;
    }

    // Calculate average
    public static double average(double[] accuracies){
        double sum = 0.0;
        for(int i = 0; i < 5; i++){
            sum += accuracies[i];
        }
        double avgAccuracy = sum / 5.0;
        return avgAccuracy;
    }

    // Calculate std. deviation
    public static double standardDeviation(double[] accuracies){
        double avgAccuracy = average(accuracies);
        double squaredSum = 0.0;
        for(int i = 0; i < 5; i++){
            squaredSum += Math.pow(accuracies[i] - avgAccuracy, 2);
        }
        double stdDev = Math.sqrt((squaredSum) / 4.0);
        return stdDev;
    }

    public static double[] confidenceInterval(double[] accuracies){
        double sum = 0.0;
        for(int i = 0; i < 5; i++){
            sum += accuracies[i];
        }
        double avgAccuracy = sum / 5.0;

        // Calculate std. deviation
        double squaredSum = 0.0;
        for(int i = 0; i < 5; i++){
            squaredSum += Math.pow(accuracies[i] - avgAccuracy, 2);
        }
        double stdDev = Math.sqrt((squaredSum) / 4.0);

        // Calculate confidence interval
        double[] confidenceIntervals = new double[2];
        confidenceIntervals[0] = avgAccuracy - 4.604 * stdDev / Math.sqrt(5.0);
        confidenceIntervals[1] = avgAccuracy + 4.604 * stdDev / Math.sqrt(5.0);

        return confidenceIntervals;
    }
}
