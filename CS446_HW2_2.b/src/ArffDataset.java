import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by nacos on 2016/9/21.
 */
public class ArffDataset {
    // Data
    final static private int INITIAL_CAPACITY = 10000;
    private int _attributesCount; // This excludes the label!
    private int _instancesCount;

    private double[][] _instanceXs;
    private double[] _instanceYs;

    public ArffDataset(final int attributesCount, final int instancesCount, final double[][] instanceXs, final double[] instanceYs){
        this._attributesCount = attributesCount;
        this._instancesCount = instancesCount;

        this._instanceXs = instanceXs;
        this._instanceYs = instanceYs;
    }

    /**
     * Factory method
     */
    public static ArffDataset readFromFile(String arffFileName) throws Exception{
        // Read from .ARFF file
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(arffFileName))
        );

        // Prepare Data
        ArrayList<double[]> instanceXList = new ArrayList<>(INITIAL_CAPACITY);
        ArrayList<Double> instanceYList = new ArrayList<>(INITIAL_CAPACITY);
        int attributesCount = 0; // This excludes the label!
        int instancesCount = 0;

        // Get attribute count
        String line;
        while((line = reader.readLine()) != null){
            if(line.startsWith("@attribute")) {
                attributesCount++;
            } else if(line.equals("@data")){
                break;
            }
        }
        attributesCount--;
        //System.out.println("Instances have " + attributesCount + " attributes with labels");

        // Get instances' X and Y
        while((line = reader.readLine()) != null) {
            double[] currentInstance = new double[attributesCount];

            int i = 0;
            for(; i < line.length()/2; i++){
                currentInstance[i] = (line.charAt(i*2) == '1'? 1 : 0);
            }

            instanceXList.add(currentInstance);
            instanceYList.add(line.charAt(i*2) == '+'? 1.0 : -1.0);

            instancesCount++;
        }
        //System.out.println("Altogether read " + instancesCount + " instances.");

        // Construct a ArffDataSet
        double[][] instanceXs = instanceXList.toArray(new double[instancesCount][attributesCount]);
        double[] instanceYs = new double[instancesCount];
        for(int i = 0; i < instancesCount; i++) {
            instanceYs[i] = instanceYList.get(i);
        }
        return new ArffDataset(attributesCount, instancesCount, instanceXs, instanceYs);
    }

    public static ArffDataset combineDatasets(ArffDataset... arffDatasets){
        assert (arffDatasets.length > 1);

        // Get total instances count
        int attributesCount = arffDatasets[0]._attributesCount;
        int instancesCount = 0;
        for(int i = 0; i < arffDatasets.length; i++){
            ArffDataset arffDataset = arffDatasets[i];
            assert(arffDataset._attributesCount == attributesCount);
            instancesCount += arffDataset._instancesCount;
        }
        double[][] instanceXs = new double[instancesCount][attributesCount];
        double[] instanceYs = new double[instancesCount];

        // Iterate copying processes
        int copiedInstancesCount = 0;
        for(int i = 0; i < arffDatasets.length; i++){
            ArffDataset arffDataset = arffDatasets[i];

            System.arraycopy(arffDataset._instanceXs, 0, instanceXs, copiedInstancesCount, arffDataset._instancesCount);
            System.arraycopy(arffDataset._instanceYs, 0, instanceYs, copiedInstancesCount, arffDataset._instancesCount);
            copiedInstancesCount += arffDataset._instancesCount;
        }
        return new ArffDataset(attributesCount, instancesCount, instanceXs, instanceYs);
    }

    /**
     * Creates the test set for one fold of a cross-validation on the dataset.
     * @param numFolds - the number of folds in the cross-validation. Must be greater than 1.
     * @param numFold - 0 for the first fold, 1 for the second, ...
     */
    public ArffDataset testCV(int numFolds, int numFold){
        if(numFolds < 2 || numFolds > _instancesCount){
            throw new IllegalArgumentException("number of folds is less than 2 or greater than the number of instances");
        }
        if(numFold < 0 || numFold >= numFolds){
            throw new IllegalArgumentException("fold number is less than 0 or greater than the number of folds");
        }

        // Construct a new ArffDataset
        int attributesCount = _attributesCount;
        int instancesCount = _instancesCount / numFolds;

        double[][] instanceXs = new double[instancesCount][attributesCount];
        double[] instanceYs = new double[instancesCount];
        System.arraycopy(_instanceXs, numFold * instancesCount, instanceXs, 0, instancesCount);
        System.arraycopy(_instanceYs, numFold * instancesCount, instanceYs, 0, instancesCount);

        return new ArffDataset(attributesCount, instancesCount, instanceXs, instanceYs);
    }

    /**
     * Creates the training set for one fold of a cross-validation on the dataset.
     * @param numFolds - the number of folds in the cross-validation. Must be greater than 1.
     * @param numFold - 0 for the first fold, 1 for the second, ...
     */
    public ArffDataset trainCV(int numFolds, int numFold){
        if(numFolds < 2 || numFolds > _instancesCount){
            throw new IllegalArgumentException("number of folds is less than 2 or greater than the number of instances");
        }
        if(numFold < 0 || numFold >= numFolds){
            throw new IllegalArgumentException("fold number is less than 0 or greater than the number of folds");
        }

        // Construct a new ArffDataset
        int attributesCount = _attributesCount;
        int testInstancesCount = _instancesCount / numFolds;
        int instancesCount = _instancesCount - testInstancesCount;

        double[][] instanceXs = new double[instancesCount][attributesCount];
        double[] instanceYs = new double[instancesCount];

        for(int i = 0; i < numFold; i++){
            System.arraycopy(_instanceXs, testInstancesCount * i, instanceXs, testInstancesCount * i, testInstancesCount);
            System.arraycopy(_instanceYs, testInstancesCount * i, instanceYs, testInstancesCount * i, testInstancesCount);
        }
        for(int i = numFold + 1; i < numFolds; i++){
            System.arraycopy(_instanceXs, testInstancesCount * i, instanceXs, testInstancesCount * (i - 1), testInstancesCount);
            System.arraycopy(_instanceYs, testInstancesCount * i, instanceYs, testInstancesCount * (i - 1), testInstancesCount);
        }

        return new ArffDataset(attributesCount, instancesCount, instanceXs, instanceYs);
    }

    public ArffDataset shuffle(double percentage){
        if(percentage >= 1 || percentage < 0){
            throw new IllegalArgumentException("percentage should be between 0 and 1");
        }

        int attributesCount = _attributesCount;
        int instancesCount = (int) (_instancesCount * percentage);
        double[][] instanceXs = new double[instancesCount][attributesCount];
        double[] instanceYs = new double[instancesCount];

        int[] instanceIsUsed = new int[_instancesCount];
        int instanceUsedCount = 0;

        Random random = new Random();
        while(instanceUsedCount < instancesCount){
            int luckyInstanceIndex = (int) (random.nextDouble() * _instancesCount);
            while(instanceIsUsed[luckyInstanceIndex] != 0){
                luckyInstanceIndex = (int) (random.nextDouble() * _instancesCount);
            }

            System.arraycopy(_instanceXs[luckyInstanceIndex], 0, instanceXs[instanceUsedCount], 0, _attributesCount);
            instanceYs[instanceUsedCount] = _instanceYs[luckyInstanceIndex];

            instanceIsUsed[luckyInstanceIndex] = 1;
            instanceUsedCount++;
        }

        return new ArffDataset(attributesCount, instancesCount, instanceXs, instanceYs);
    }

    /**
     * Getters
     */
    public int getAttributesCount() {
        return _attributesCount;
    }
    public int getInstancesCount() {
        return _instancesCount;
    }
    public double[][] getInstanceXs() {
        return _instanceXs;
    }
    public double[] getInstanceYs() {
        return _instanceYs;
    }
}
