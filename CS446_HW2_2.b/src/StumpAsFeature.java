import weka.classifiers.Evaluation;
import weka.core.Debug;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by nacos on 2016/9/23.
 */
public class StumpAsFeature {
    // Data
    private Instances _trainDataset;
    private int _trainInstanceCount;
    private Instances _testDataset;
    private int _testInstanceCount;

    // Configuration
    final private int STUMP_COUNT = 100;

    // Result
    private Id3[] _classifiers;
    private ArffDataset _newTrainDataset;
    private ArffDataset _newTestDataset;

    public StumpAsFeature(Instances trainDataset, Instances testDataset){
        this._trainDataset = trainDataset;
        this._trainInstanceCount = trainDataset.numInstances();
        this._testDataset = testDataset;
        this._testInstanceCount = testDataset.numInstances();
    }

    public ArffDataset getStumpFeaturedTrainDataset() throws Exception{
        // Build 100 stumps
        _classifiers = new Id3[STUMP_COUNT];
        Random random = new Debug.Random();
        for(int i = 0; i < STUMP_COUNT; i++){
            // Generate subTrainDataset of 50%
            Instances subTrainDataset = _trainDataset.resample(random);
            for(int j = 0; j < _trainInstanceCount / 2; j++){
                subTrainDataset.delete(j);
            }

            // Use ID3 to build classifier with given depth
            Id3 classifier = new Id3();
            classifier.setMaxDepth(4);
            classifier.buildClassifier(subTrainDataset);

            _classifiers[i] = classifier;
        }

        // Generate new dataset
        double[][] instanceXs = new double[_trainInstanceCount][STUMP_COUNT];
        double[] instanceYs = new double[_trainInstanceCount];

        for(int i = 0; i < _trainInstanceCount; i++){
            instanceXs[i] = new double[STUMP_COUNT];
            for(int j = 0; j < STUMP_COUNT; j++) {

                double prediction = _classifiers[j].classifyInstance(_trainDataset.instance(i));
                instanceXs[i][j] = prediction < 0.5? 1.0 : -1.0;
            }

            instanceYs[i] = _trainDataset.instance(i).classValue() < 0.5? 1.0 : -1.0;
        }

        _newTrainDataset = new ArffDataset(STUMP_COUNT, _trainInstanceCount, instanceXs, instanceYs);
        return _newTrainDataset;
    }

    public ArffDataset getStumpFeaturedTestDataset() throws Exception{
        if(_classifiers == null){
            throw new NullPointerException("Id3 classifiers not ready");
        }

        // Generate new dataset
        double[][] instanceXs = new double[_testInstanceCount][STUMP_COUNT];
        double[] instanceYs = new double[_testInstanceCount];

        for(int i = 0; i < _testInstanceCount; i++){
            instanceXs[i] = new double[STUMP_COUNT];
            for(int j = 0; j < STUMP_COUNT; j++) {

                double prediction = _classifiers[j].classifyInstance(_testDataset.instance(i));
                instanceXs[i][j] = prediction < 0.5? 1.0 : -1.0;
            }

            instanceYs[i] = _testDataset.instance(i).classValue() < 0.5? 1.0 : -1.0;
        }

        _newTestDataset = new ArffDataset(STUMP_COUNT, _testInstanceCount, instanceXs, instanceYs);
        return _newTestDataset;
    }
}
