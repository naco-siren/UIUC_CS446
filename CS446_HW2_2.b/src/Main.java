import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws Exception {
        ReportingModule.do5FoldCrossValidation();
    }
}
