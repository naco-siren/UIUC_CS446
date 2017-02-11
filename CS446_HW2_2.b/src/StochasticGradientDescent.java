import java.util.ArrayList;
import java.util.Random;

/**
 * Created by nacos on 2016/9/19.
 */
public class StochasticGradientDescent {
    // Configuration
    private double _errorThreshold;
    private double _learningRate;
    final private int VALIDATION_FOLD = 5;

    // Data
    private int _attributesCount; // This excludes the label!
    private int _instancesCount;

    private double[][] _instanceXs;
    private double[] _instanceYs;

    // Result
    private double[] _weights;

    public StochasticGradientDescent(final ArffDataset arffDataset){
        this._attributesCount = arffDataset.getAttributesCount();
        this._instancesCount = arffDataset.getInstancesCount();

        this._instanceXs = arffDataset.getInstanceXs();
        this._instanceYs = arffDataset.getInstanceYs();
    }

    public int buildClassifier() throws Exception{
        // Prepare validation
        int availableTrainInstanceCount = _instancesCount;
        int validationInstanceCount = 0;
        int[] validationFlags = new int[_instancesCount];
        for(int i = _instancesCount * (VALIDATION_FOLD - 1) / VALIDATION_FOLD; i < _instancesCount; i++){
            validationFlags[i] = 1;
            availableTrainInstanceCount--;
            validationInstanceCount++;
        }

        // Prepare flags: 0 - buildClassifier, 1 - validation, -1 - used.
        int[] instanceFlags = new int[_instancesCount];
        for(int i = 0; i < _instancesCount; i++){
            instanceFlags[i] = validationFlags[i];
        }

        // Initialize learning conditions
        _weights = new double[_attributesCount];
        _learningRate = 0.06;

        // Loop to update weights
        double error = 100;
        _errorThreshold = 0.36 * validationInstanceCount; // TODO: error threshold may need tuning
        Random random = new Random();
        while(error > _errorThreshold && availableTrainInstanceCount > 0){
            // Calculate the error to see if convergence
            error = calculateErrorOnValidation(_weights, _instanceXs, _instanceYs, instanceFlags);
            System.out.println("Error: " + error);

            // Randomly select a buildClassifier instance to update weights
            int nextTrainIndex = random.nextInt(_instancesCount);
            while(instanceFlags[nextTrainIndex] != 0){
                nextTrainIndex = random.nextInt(_instancesCount);
            }
            double[] currentInstanceX = _instanceXs[nextTrainIndex];
            double currentInstanceY = _instanceYs[nextTrainIndex];

            // Update the weights
            for(int i = 0; i < _attributesCount; i++){
                double currentPrediction = VectorUtil.doInnerProduct(_weights, currentInstanceX);
                _weights[i] += _learningRate * (currentInstanceY - currentPrediction) * currentInstanceX[i];
            }

            // Mark this instance as learned
            instanceFlags[nextTrainIndex] = -1;
            availableTrainInstanceCount--;
        }
        System.out.println("SGD training finished.");
        return 0;
    }

    public double calculateErrorOnValidation(double[] weights, double[][] instanceXs, double[] instanceYs, int[] instanceFlags){
        assert (instanceXs.length == instanceYs.length);
        assert (instanceXs.length == instanceFlags.length);

        double error = 0.0;
        for (int i = 0; i < instanceFlags.length; i++){
            if(instanceFlags[i] == 1){
                double product = VectorUtil.doInnerProduct(weights, instanceXs[i]);
                error += Math.pow(instanceYs[i] - product, 2);
            }
        }
        error /= 2.0;

        return error;
    }

    public double accuracyRateOn(ArffDataset testDataset){
        int testInstanceCount = testDataset.getInstancesCount();
        int correctPredictionCount = 0;

        double[][] testInstanceXs = testDataset.getInstanceXs();
        double[] testInstanceYs = testDataset.getInstanceYs();

        for(int i = 0; i < testInstanceCount; i++){
            if(VectorUtil.predictLabel(_weights, testInstanceXs[i]) == testInstanceYs[i]){
                correctPredictionCount++;
            }
        }

        double accuracy = correctPredictionCount / (testInstanceCount * 1.0);
        return accuracy;
    }

    public double[] getWeights() {
        return _weights;
    }
}
