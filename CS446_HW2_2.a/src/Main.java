/**
 * Created by nacos on 2016/9/18.
 */



public class Main {

    public static void main(String[] args) throws Exception{

        // Generate .ARFF file
        String filename = "./badges/badges.modified.data.all";
        FeatureExtraction featureExtraction = new FeatureExtraction(filename);
        featureExtraction.extractToFile("./badges.arff");

        // Generate folded .ARFF files
        filename = "./badges/badges.modified.data.fold";
        for(int i = 0; i < 5; i++) {
            featureExtraction = new FeatureExtraction(filename + (i+1));
            featureExtraction.extractToFile("./badges/badges.modified.data.fold" + (i+1) + ".arff");
        }

    }
}
